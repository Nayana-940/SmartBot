import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from "dotenv";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import readline from 'readline';

// ✅ Load environment variables
dotenv.config();

(async () => {
  try {
    console.log("🚀 Connecting to Pinecone...");

    // ✅ Initialize Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    console.log("✅ Connected to Pinecone");

    // ✅ Connect to the index
    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // ✅ Use Google's Gemini API for embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: "text-embedding-004",
      taskType: "RETRIEVAL_DOCUMENT",
    });

    // ✅ Initialize PineconeStore from the existing index
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
    });

    console.log("✅ PineconeStore initialized");

    // ✅ Use ChatGoogleGenerativeAI from LangChain
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-2.0-flash",
    });

    console.log("✅ Gemini AI Model Loaded");

    // ✅ Function to query Pinecone and get AI-generated answers
    const queryAndGenerateResponse = async (query) => {
      console.log(`🔍 Searching for: "${query}"...`);

      // ✅ Perform similarity search in Pinecone
      const retrievedDocs = await vectorStore.similaritySearch(query, 5);

      if (retrievedDocs.length === 0) {
        return "❌ No relevant information found in the knowledge base.";
      }

      // ✅ Combine document content with emphasis on location
      const docsContent = retrievedDocs.map(doc => doc.pageContent).join("\n");

      // ✅ Generate response using Gemini with specific location prompt
      const fullPrompt = `Answer the following question based on the provided context. Focus on location details if relevant.\n\nContext:\n${docsContent}\n\nQuestion: ${query}`;

      const result = await model.invoke(fullPrompt);
      return result.text;
    };

    // ✅ Interactive Loop for Questions
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    function askQuestion(query) {
      return new Promise(resolve => rl.question(query, resolve));
    }

    let query = await askQuestion("💬 Ask your question (type 'exit' to quit): ");
    while (query.toLowerCase() !== "exit") {
      const answer = await queryAndGenerateResponse(query);
      console.log("🤖 Answer:", answer);

      query = await askQuestion("\n💬 Ask another question (or type 'exit' to quit): ");
    }

    rl.close();
    console.log("👋 Chatbot session ended.");

  } catch (error) {
    console.error("❌ Error:", error.message);
  }
})();
