import { Request, Response } from 'express';
import {
  BatchJobResults,
  BatchJobStatus,
  BatchPredictionItem,
  BatchRequest,
  HealthResponse,
  PredictRequest,
} from '../src/api/types';
import { mockPredict } from './data';

const EXPECTED_API_KEY = process.env.MOCK_API_KEY || process.env.API_KEY || 'test-key-dev';

export function authMiddleware(req: Request, res: Response, next: () => void): void {
  // Public endpoint
  if (req.path === '/health') {
    next();
    return;
  }

  const apiKeyHeader = req.headers['x-api-key'];
  let token: string | undefined;

  if (typeof apiKeyHeader === 'string') {
    token = apiKeyHeader.trim();
  } else if (req.headers.authorization?.startsWith('Bearer ')) {
    token = req.headers.authorization.slice(7).trim();
  }

  const isValid =
    Boolean(token) &&
    (token === EXPECTED_API_KEY ||
      token === 'test-key-dev' ||
      token?.startsWith('tf2_') ||
      (process.env.API_KEY && token === process.env.API_KEY));

  if (!isValid) {
    res.status(401).json({
      error: {
        code: 'UNAUTHORIZED',
        message: 'Invalid or missing API key. Provide X-API-Key header or Bearer token.',
      },
    });
    return;
  }

  next();
}

export function handleHealth(_req: Request, res: Response): void {
  const response: HealthResponse = {
    status: 'ok',
    model_version: 'v2.0.0-phase2-prod',
    model_loaded: true,
    device: 'cpu',
  };
  res.json(response);
}

export function handlePredict(req: Request, res: Response): void {
  const body = req.body as PredictRequest;

  if (!body || typeof body.text !== 'string' || body.text.trim().length === 0) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Validation failed for predict request',
        details: [{ field: 'text', issue: 'Field "text" is required and cannot be empty' }],
      },
    });
    return;
  }

  const validChannels = ['email', 'chat', 'call_transcript'];
  if (!body.channel || !validChannels.includes(body.channel)) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Invalid channel specified',
        details: [
          {
            field: 'channel',
            issue: `Channel must be one of: ${validChannels.join(', ')}`,
          },
        ],
      },
    });
    return;
  }

  const prediction = mockPredict(body);
  res.json(prediction);
}

export function handlePredictBatch(req: Request, res: Response): void {
  const body = req.body as BatchRequest;

  if (!body || !Array.isArray(body.tickets)) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Body must contain "tickets" array',
        details: [{ field: 'tickets', issue: 'Field "tickets" is required and must be an array' }],
      },
    });
    return;
  }

  if (body.tickets.length === 0) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Batch cannot be empty',
        details: [{ field: 'tickets', issue: 'Must provide at least 1 ticket' }],
      },
    });
    return;
  }

  if (body.tickets.length > 100) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Batch size exceeds maximum limit of 100 tickets',
        details: [{ field: 'tickets', issue: 'Batch size cannot exceed 100 tickets' }],
      },
    });
    return;
  }

  const start = performance.now();
  const predictions: BatchPredictionItem[] = body.tickets.map((t, idx) => {
    const tid = t.ticket_id || `TF-BATCH-${String(idx + 1).padStart(3, '0')}`;
    return {
      ...mockPredict(t),
      ticket_id: tid,
    };
  });
  const duration = Math.round(performance.now() - start);

  res.json({
    predictions,
    meta: {
      count: predictions.length,
      model_version: 'v2.0.0-phase2-prod',
      processing_time_ms: duration,
    },
  });
}

// In-memory jobs store for mock async jobs
interface StoredJob {
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  total: number;
  processed: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  predictions: BatchPredictionItem[];
  simulationTimer?: NodeJS.Timeout;
}

const jobsStore = new Map<string, StoredJob>();

export function handleCreateJob(req: Request, res: Response): void {
  const body = req.body as BatchRequest;

  if (!body || !Array.isArray(body.tickets) || body.tickets.length === 0) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Tickets array required',
        details: [{ field: 'tickets', issue: 'Tickets array is required and cannot be empty' }],
      },
    });
    return;
  }

  if (body.tickets.length > 5000) {
    res.status(422).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Job size exceeds maximum limit of 5,000 tickets',
        details: [{ field: 'tickets', issue: 'Maximum tickets allowed is 5000' }],
      },
    });
    return;
  }

  const jobId = `job_${Math.random().toString(36).slice(2, 10)}`;
  const now = new Date().toISOString();

  const predictions: BatchPredictionItem[] = body.tickets.map((t, i) => ({
    ...mockPredict(t),
    ticket_id: t.ticket_id || `TF-JOB-${String(i + 1).padStart(4, '0')}`,
  }));

  const job: StoredJob = {
    job_id: jobId,
    status: 'queued',
    total: body.tickets.length,
    processed: 0,
    created_at: now,
    started_at: null,
    finished_at: null,
    predictions,
  };

  // Simulate progress
  const stepInterval = 800;
  const steps = 6;
  const increment = Math.ceil(job.total / steps);

  job.simulationTimer = setInterval(() => {
    if (job.status === 'cancelled') {
      if (job.simulationTimer) clearInterval(job.simulationTimer);
      return;
    }

    if (job.status === 'queued') {
      job.status = 'running';
      job.started_at = new Date().toISOString();
    }

    job.processed = Math.min(job.total, job.processed + increment);

    if (job.processed >= job.total) {
      job.status = 'succeeded';
      job.finished_at = new Date().toISOString();
      if (job.simulationTimer) clearInterval(job.simulationTimer);
    }
  }, stepInterval);

  jobsStore.set(jobId, job);

  const statusResponse: BatchJobStatus = {
    job_id: job.job_id,
    status: job.status,
    total: job.total,
    processed: job.processed,
    created_at: job.created_at,
    model_version: 'v2.0.0-phase2-prod',
  };

  res.status(202).json(statusResponse);
}

export function handleGetJobStatus(req: Request, res: Response): void {
  const jobId = String(req.params.job_id);
  const job = jobsStore.get(jobId);

  if (!job) {
    res.status(404).json({
      error: {
        code: 'NOT_FOUND',
        message: `Batch job ${jobId} not found`,
      },
    });
    return;
  }

  // Add Retry-After header while job is pending
  if (job.status === 'queued' || job.status === 'running') {
    res.setHeader('Retry-After', '2');
  }

  const response: BatchJobStatus = {
    job_id: job.job_id,
    status: job.status,
    total: job.total,
    processed: job.processed,
    created_at: job.created_at,
    started_at: job.started_at,
    finished_at: job.finished_at,
    model_version: 'v2.0.0-phase2-prod',
  };

  res.json(response);
}

export function handleGetJobResults(req: Request, res: Response): void {
  const jobId = String(req.params.job_id);
  const job = jobsStore.get(jobId);

  if (!job) {
    res.status(404).json({
      error: {
        code: 'NOT_FOUND',
        message: `Batch job ${jobId} not found`,
      },
    });
    return;
  }

  if (job.status !== 'succeeded') {
    res.status(409).json({
      error: {
        code: 'JOB_NOT_READY',
        message: `Job is currently in '${job.status}' status. Results are only available when succeeded.`,
      },
    });
    return;
  }

  const offset = parseInt(req.query.offset as string, 10) || 0;
  const limit = parseInt(req.query.limit as string, 10) || 500;

  const slice = job.predictions.slice(offset, offset + limit);
  const nextOffset = offset + limit < job.total ? offset + limit : null;

  const response: BatchJobResults = {
    job_id: job.job_id,
    status: 'succeeded',
    total: job.total,
    offset,
    limit,
    next_offset: nextOffset,
    model_version: 'v2.0.0-phase2-prod',
    predictions: slice,
  };

  res.json(response);
}

export function handleCancelJob(req: Request, res: Response): void {
  const jobId = String(req.params.job_id);
  const job = jobsStore.get(jobId);

  if (!job) {
    res.status(404).json({
      error: {
        code: 'NOT_FOUND',
        message: `Batch job ${jobId} not found`,
      },
    });
    return;
  }

  if (job.status === 'succeeded' || job.status === 'failed') {
    res.status(400).json({
      error: {
        code: 'CANNOT_CANCEL',
        message: `Job already reached terminal state '${job.status}'`,
      },
    });
    return;
  }

  job.status = 'cancelled';
  if (job.simulationTimer) clearInterval(job.simulationTimer);

  const response: BatchJobStatus = {
    job_id: job.job_id,
    status: 'cancelled',
    total: job.total,
    processed: job.processed,
    created_at: job.created_at,
    finished_at: new Date().toISOString(),
    model_version: 'v2.0.0-phase2-prod',
  };

  res.json(response);
}
