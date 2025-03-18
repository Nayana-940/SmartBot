import dotenv from "dotenv";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import promptSync from "prompt-sync";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// ✅ Load environment variables
dotenv.config();
const prompt = promptSync();

// ✅ Initialize Pinecone
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
console.log("✅ Connected to Pinecone");

// ✅ Connect to Pinecone index
const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// ✅ Use Google's Gemini API for embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  taskType: "RETRIEVAL_DOCUMENT",
});

// ✅ Initialize Pinecone Vector Store from existing index
const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex: index,
});
console.log("✅ Pinecone Vector Store Initialized");

// ✅ Initialize Chat Model using LangChain
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "gemini-2.0-flash",
});
console.log("✅ Gemini AI Model Loaded");

// ✅ Define a Prompt Template for Chat
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "Answer the user's question based on the context below:\n\n{context}"],
  ["user", "{question}"],
]);

// ✅ Function to Retrieve Relevant Data from Pinecone
const retrieve = async (query) => {
  console.log(`🔍 Searching for: "${query}"...`);
  const retrievedDocs = await vectorStore.similaritySearch(query, 5);

  if (retrievedDocs.length === 0) {
    console.log("❌ No relevant information found.");
    return "";
  }

  return retrievedDocs.map(doc => doc.pageContent).join("\n");
};

// ✅ Function to Generate Answer Using Gemini
const generateAnswer = async (query, context) => {
  const formattedPrompt = await promptTemplate.format({
    question: query,
    context: context,
  });

  const result = await model.invoke(formattedPrompt);
  return result.content;
};

// ✅ Interactive Chat Loop
let query = prompt("💬 Ask your question (type 'exit' to quit): ");
while (query.toLowerCase() !== "exit") {
  const context = await retrieve(query);
  console.log("Retrieved context is:", context)

  if (!context) {
    console.log("🤖 Answer: Sorry, I couldn't find any relevant information.");
  } else {
    const answer = await generateAnswer(query, context);
    console.log("🤖 Answer:", answer);
  }

  query = prompt("\n💬 Ask another question (or type 'exit' to quit): ");
}

console.log("👋 Chatbot session ended.");
