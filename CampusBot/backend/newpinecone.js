import dotenv from "dotenv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { PlaywrightWebBaseLoader } from "@langchain/community/document_loaders/web/playwright"; // Fixed import

dotenv.config({ path: "./.env" });

async function main() {
  try {
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    console.log("Connected to Pinecone");

    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    const embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "embedding-001",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    });

    const websiteUrls =[
      "https://mgmits.ac.in/b-tech-admissions-2021/",
    "https://mgmits.ac.in/mcaadmissions2022/",
    "https://mgmits.ac.in/m-tech-admissions-2023/",
    "https://mgmits.ac.in/contact-us/"
    ]
    
    
    
    
    
    
    
    
    
    

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    let allSplits = [];

    for (const websiteUrl of websiteUrls) {
      console.log(`Loading website content from: ${websiteUrl}`);

      try {
        const loader = new PlaywrightWebBaseLoader(websiteUrl, {
          launchOptions: { headless: true }, // Ensure Playwright runs headless
          timeout: 30000,
        });

        const docs = await loader.load();

        docs.forEach(doc => {
          doc.pageContent = doc.pageContent.replace(/\s+/g, ' ').trim();
          doc.metadata = {
            ...doc.metadata,
            source: websiteUrl,
            title: websiteUrl.split('/').pop() || 'MITS Page'
          };
        });

        console.log(`Loaded content from ${websiteUrl}`);
        console.log(`Content length: ${docs[0].pageContent.length} characters`);

        const splits = await splitter.splitDocuments(docs);
        console.log(`Split into ${splits.length} chunks`);

        allSplits = allSplits.concat(splits);
      } catch (error) {
        console.error(`Error loading ${websiteUrl}:`, error);
        continue;
      }
    }

    console.log(`Total chunks across all URLs: ${allSplits.length}`);

    if (allSplits.length === 0) {
      throw new Error("No content was loaded from any URLs");
    }

    const sampleEmbedding = await embeddings.embedQuery(allSplits[0].pageContent);
    console.log(`Sample embedding dimension: ${sampleEmbedding.length}`);

    const vectorStore = await PineconeStore.fromDocuments(
      allSplits,
      embeddings,
      { pineconeIndex: index }
    );

    console.log("Website content indexed successfully!");
  } catch (error) {
    console.error("Error during execution:", error);
  }
}

main().catch(console.error);
