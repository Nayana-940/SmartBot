# MITS Campus Assistant

A conversational AI chatbot designed to provide information about Muthoot Institute of Technology & Science (MITS). The assistant uses Pinecone for vector storage and retrieval, Google's Gemini Pro for natural language understanding, and offers a modern, responsive web interface.

## Features

- Real-time chat interface with typing indicators
- Semantic search using Pinecone vector database
- Context-aware responses using Google's Gemini Pro
- Modern, responsive UI design
- Dynamic port detection for both frontend and backend
- Comprehensive error handling and user feedback

## Prerequisites

- Node.js v16 or higher
- npm (Node Package Manager)
- Google API Key (for Gemini Pro)
- Pinecone API Key and Index

## Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
GOOGLE_API_KEY=your_google_api_key
```

## Installation

1. Clone the repository
2. Install backend dependencies:
   ```bash
   cd backend
   npm install
   ```
3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```bash
   cd backend
   node server.js
   ```
   The server will automatically find an available port.

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```
   The frontend will run on port 3000 by default.

## Project Structure

```
├── backend/
│   ├── server.js          # Express server and API endpoints
│   ├── config.js          # Server configuration
│   └── package.json       # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   ├── App.css       # Styles
│   │   └── config.js     # Frontend configuration
│   └── package.json      # Frontend dependencies
└── README.md             # Project documentation
```

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /chat` - Chat endpoint for sending messages and receiving responses

## Technologies Used

- **Frontend**:
  - React
  - Modern CSS with CSS Variables
  - Responsive Design

- **Backend**:
  - Express.js
  - LangChain
  - Google Generative AI (Gemini Pro)
  - Pinecone Vector Database

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Muthoot Institute of Technology & Science
- Pinecone for vector database capabilities
- Google for Generative AI services
