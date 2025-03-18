import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import config from './config.js';
import { promises as fs } from 'fs';

dotenv.config();
const app = express();

// Configure CORS
app.use(cors());

app.use(express.json());

// Initialize Pinecone client
const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
});

// Initialize embeddings model
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    modelName: "embedding-001"
});

// Initialize LLM
const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY,
    modelName: "gemini-2.0-flash",
    maxOutputTokens: 2048,
    temperature: 0.7
});

let vectorStore;

const initVectorStore = async () => {
    try {
        const index = pc.Index(process.env.PINECONE_INDEX_NAME);
        vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });
        console.log("✅ Vector store initialized successfully");
    } catch (error) {
        console.error("❌ Failed to initialize vector store:", error);
        throw error;
    }
};

app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        timestamp: new Date().toISOString()
    });
});

const generateMITSResponse = async (question, context) => {
    const systemPrompt = `You are the MITS (Muthoot Institute of Technology & Science) Campus Assistant, a helpful AI designed to assist students, faculty, and visitors with information about MITS. 

Key points about your role:
1. Be professional, friendly, and concise
2. Focus on providing accurate information about Muthoot Institute of Technology & Science
3. If you're not sure about something, admit it and suggest contacting the relevant department
4. For questions about courses, admissions, or departments, provide official contact information
5. Maintain a helpful and encouraging tone

Based on this context: "${context}"

Question: "${question}"

Provide a clear, helpful response focusing on MITS-specific information. If the context doesn't contain enough information, say you don't have specific information about that aspect of MITS and suggest where they might find the information.`;

    try {
        console.log("System prompt", systemPrompt)
        const response = await llm.invoke(systemPrompt);
        return response.content;
    } catch (error) {
        console.error("Error generating response:", error);
        throw error;
    }
};


app.post("/chat", async (req, res) => {
    console.log("Entering chat")
    try {
        const { message } = req.body;
        if (!message || !message.trim()) {
            return res.status(400).json({ error: 'Please provide a valid question' });
        }

        if (!vectorStore) {
            return res.status(503).json({ error: 'Vector store not initialized' });
        }

        console.log("looking up vector store")
        
        // Enhance search queries for leadership-related questions
        let enhancedQuery = message;
        const leadershipTerms = ['principal', 'vice principal', 'dean', 'director', 'head'];
        
        if (leadershipTerms.some(term => message.toLowerCase().includes(term))) {
            enhancedQuery = `${message} MITS leadership administration management executive-body`;
        }

        // Increase results for better context
        const results = await vectorStore.similaritySearch(enhancedQuery, 5);
        console.log("received results", results.length)
        
        if (!results || results.length === 0) {
            return res.json({ 
                response: "I don't have specific information about that aspect of Muthoot Institute of Technology & Science. I recommend checking the official MITS website (www.mgits.ac.in) or contacting the relevant department for the most accurate information.",
                timestamp: new Date().toISOString()
            });
        }

        // Sort results by relevance for leadership queries
        if (leadershipTerms.some(term => message.toLowerCase().includes(term))) {
            results.sort((a, b) => {
                const aRelevance = leadershipTerms.reduce((score, term) => 
                    score + (a.pageContent.toLowerCase().includes(term) ? 1 : 0), 0);
                const bRelevance = leadershipTerms.reduce((score, term) => 
                    score + (b.pageContent.toLowerCase().includes(term) ? 1 : 0), 0);
                return bRelevance - aRelevance;
            });
        }

        const context = results.map(r => r.pageContent).join('\n\n');
        console.log("Message:", message)
        console.log("Enhanced Query:", enhancedQuery)
        console.log("Context Length:", context.length)

        const response = await generateMITSResponse(message, context);
        console.log("response generated")

        res.json({
            response: response,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('Error processing message:', error);
        res.status(500).json({ 
            error: 'I encountered an error while processing your question. Please try rephrasing it or contact MITS support for assistance.',
            timestamp: new Date().toISOString()
        });
    }
});

console.log(config.basePort);

const findAvailablePort = async (startPort, maxTries) => {
    for (let port = startPort; port < startPort + maxTries; port++) {
        try {
            const server = await new Promise((resolve, reject) => {
                const srv = app.listen(port)
                    .once('listening', () => {
                        resolve(srv);
                    })
                    .once('error', (err) => {
                        if (err.code === 'EADDRINUSE') {
                            resolve(false);
                        } else {
                            reject(err);
                        }
                    });
            });

            if (server) {
                return { server, port };
            }
        } catch (error) {
            console.error(`Error trying port ${port}:`, error);
        }
    }
    throw new Error(`No available ports found after ${maxTries} attempts`);
};
console.log(config.basePort);

const startServer = async () => {
    try {
        // Initialize vector store first
        await initVectorStore();
        
        // Find an available port
        const { server, port } = await findAvailablePort(config.basePort, config.maxPortTries);
        
        console.log(`🚀 Server running on port ${port}`);
        console.log(`💻 Health check: http://localhost:${port}/health`);
        console.log(`🤖 MITS Campus Assistant is ready to help!`);
        
        // Write the active port to a file that the frontend can read
        await fs.writeFile('./active-port.json', JSON.stringify({ port }));

        // Handle server errors
        server.on('error', (error) => {
            console.error('❌ Server error:', error);
            process.exit(1);
        });

    } catch (error) {
        console.error('❌ Failed to start server:', error);
        process.exit(1);
    }
};

startServer();
