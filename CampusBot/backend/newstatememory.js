import dotenv from "dotenv";
import promptSync from "prompt-sync"; 
import { pull } from "langchain/hub"; 
import { Annotation, StateGraph } from "@langchain/langgraph"; 
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"; 
import { TaskType } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone"; 
import { PineconeStore } from "@langchain/pinecone"; 

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

// ✅ Use Gemini API for Generating Responses
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0.7
});

// ✅ Define state for application (add history)
const StateAnnotation = Annotation.Root({
  question: Annotation(),
  context: Annotation(),
  answer: Annotation(),
  history: Annotation()
});

// ✅ Function to retrieve context using history
const retrieve = async (state) => {
  console.log("🔍 Retrieving context...");

  // Combine last AI response with the new question to refine search
  const searchQuery = state.history.length
    ? `${state.history[state.history.length - 1].ai} ${state.question}`
    : state.question;

  const retrievedDocs = await vectorStore.similaritySearch(searchQuery);
  return { context: retrievedDocs.map(doc => doc.pageContent).join("\n") };
};

// ✅ Function to generate answer using context and history
const generate = async (state) => {
  console.log("💡 Generating response...");

  const contextWithHistory = state.history
    .map(entry => `Human: ${entry.human}\nAI: ${entry.ai}`)
    .join("\n");

  const fullContext = `${contextWithHistory}\nContext: ${state.context}`;
  
  const messages = await promptTemplate.invoke({ 
    question: state.question, 
    context: fullContext 
  });

  const response = await llm.invoke(messages);

  return { 
    answer: response.content,
    history: [
      ...state.history,
      { human: state.question, ai: response.content }
    ]
  };
};

// ✅ Compile application steps
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

// ✅ Main function to handle user input
const chatLoop = async () => {
  console.log("✅ Chatbot Initialized\n");

  let state = {
    question: "",
    context: "",
    answer: "",
    history: []
  };

  let query = null;

  while (true) {
    query = prompt("💬 Ask about MITS: ");
    if (query.toLowerCase() === "exit") break;

    state = await graph.invoke({
      ...state,
      question: query
    });

    console.log(`🤖 AI: ${state.answer}\n`);
  }

  console.log("\n📚 Conversation History:");
  state.history.forEach(entry => {
    console.log(`👤 Human: ${entry.human}`);
    console.log(`🤖 AI: ${entry.ai}\n`);
  });
};

// ✅ Start chat loop
chatLoop();
