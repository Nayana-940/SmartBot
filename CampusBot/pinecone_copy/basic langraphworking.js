import dotenv from "dotenv";
import promptSync from "prompt-sync"; 
import { SitemapLoader } from "@langchain/community/document_loaders/web/sitemap"; 
import { ChatPromptTemplate } from "@langchain/core/prompts"; 
import { pull } from "langchain/hub"; 
import { Annotation, StateGraph } from "@langchain/langgraph"; 
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"; 
import { TaskType } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone"; // ✅ Import Pinecone
import { PineconeStore } from "@langchain/pinecone"; // ✅ Langchain integration for Pinecone

// Load environment variables
dotenv.config();
const prompt = promptSync();

// ✅ Initialize Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// ✅ Use Gemini API for Embeddings
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "text-embedding-004",
  taskType: TaskType.RETRIEVAL_DOCUMENT
});

// ✅ Use Pinecone Vector Store
const vectorStore = new PineconeStore(embeddings, { pineconeIndex: index });

// ✅ Define prompt for question-answering
const promptTemplate = await pull("rlm/rag-prompt");

const retrieve = async (state) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  return { context: retrievedDocs };
};

// ✅ Use Gemini API for Generating Responses
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7
});

const generate = async (state) => {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};

// ✅ Define state for application
const StateAnnotation = Annotation.Root({
  question: Annotation(),
  context: Annotation(),
  answer: Annotation(),
});

// ✅ Compile application steps
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

// ✅ Main function to handle user input
let query = null;

do {
  query = prompt("Enter your question. Type 'exit' to quit: ") ?? "exit";
  if (query !== "exit") {
    let inputs = { question: query };
    const result = await graph.invoke(inputs);
    console.log(result.answer);
  }
} while (query !== "exit");





