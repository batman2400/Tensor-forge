import express from 'express';
import cors from 'cors';
import {
  authMiddleware,
  handleCancelJob,
  handleCreateJob,
  handleGetJobResults,
  handleGetJobStatus,
  handleHealth,
  handlePredict,
  handlePredictBatch,
} from './handlers';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Health check is public
app.get('/health', handleHealth);

// Authenticated endpoints
app.use(authMiddleware);

app.post('/predict', handlePredict);
app.post('/predict/batch', handlePredictBatch);
app.post('/batch/jobs', handleCreateJob);
app.get('/batch/jobs/:job_id', handleGetJobStatus);
app.get('/batch/jobs/:job_id/results', handleGetJobResults);
app.delete('/batch/jobs/:job_id', handleCancelJob);

app.listen(PORT, () => {
  console.log(`[Mock Server] RideEat TensorForge Mock API listening on port ${PORT}`);
  console.log(`[Mock Server] Endpoints:`);
  console.log(`  - GET    /health (public)`);
  console.log(`  - POST   /predict (auth required)`);
  console.log(`  - POST   /predict/batch (auth required)`);
  console.log(`  - POST   /batch/jobs (auth required)`);
  console.log(`  - GET    /batch/jobs/:job_id (auth required)`);
  console.log(`  - GET    /batch/jobs/:job_id/results (auth required)`);
  console.log(`  - DELETE /batch/jobs/:job_id (auth required)`);
  console.log(`[Mock Server] Default Dev API Key: "test-key-dev"`);
});
