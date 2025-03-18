import dotenv from "dotenv";
import { SitemapLoader } from "@langchain/community/document_loaders/web/sitemap";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone"; // Updated import
import { PineconeStore } from "@langchain/pinecone";

// ✅ Load environment variables
dotenv.config({ path: './.env' });

async function main() {
  // ✅ Initialize Pinecone
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY, // Only the API key is required
  });

  console.log("✅ Connected to Pinecone");

  // ✅ Connect to your index
  const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  // ✅ Use Gemini API for Embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004",
    taskType: TaskType.RETRIEVAL_DOCUMENT
  });

  // ✅ Load data from sitemap
  const sitemapLoader = new SitemapLoader("https://mgmits.ac.in/post-sitemap2.xml", {
    maxConcurrency: 2,
    timeout: 120000,
  });

  console.log("🔄 Loading documents from sitemap...");
  const docs = await sitemapLoader.load();
  console.log(`✅ Loaded ${docs.length} documents`);

  // ✅ Split text into smaller chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
  });

  const allSplits = await splitter.splitDocuments(docs);
  console.log(`✅ Split into ${allSplits.length} chunks`);

  // ✅ Store extracted and chunked data in Pinecone
  const vectorStore = await PineconeStore.fromDocuments(
    allSplits,
    embeddings,
    { pineconeIndex: index }
  );

  console.log("✅ Sitemap content indexed successfully!");
}

main().catch(console.error);