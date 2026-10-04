/**
 * Type definitions strictly derived from the OpenAPI specification
 * (`tensorforge-phase2-openapi-v2.yaml`) for RideEat TensorForge 2.0.
 */

export type Channel = 'email' | 'chat' | 'call_transcript';

export type Category =
  | 'payment_refund'
  | 'ride_trip_issue'
  | 'lost_item'
  | 'order_missing_wrong'
  | 'delivery_delay'
  | 'food_quality'
  | 'account_promo'
  | 'safety_conduct'
  | 'app_technical'
  | 'general_inquiry'
  | 'spam_irrelevant';

export type Team =
  | 'Payments & Refunds'
  | 'Ride Operations'
  | 'Lost & Found'
  | 'Food Operations'
  | 'Delivery Operations'
  | 'Restaurant Quality'
  | 'Account Services'
  | 'Trust & Safety'
  | 'Tech Support'
  | 'Front-line Support'
  | 'Auto-close / Spam Filter';

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface PredictRequest {
  ticket_id?: string;
  channel: Channel;
  subject?: string;
  text: string;
}

export interface PredictResponse {
  ticket_id?: string;
  category: Category;
  secondary_category: Category | null;
  team: Team;
  is_urgent: boolean;
  confidence: number;
  model_version: string;
  needs_human_review?: boolean;
  latency_ms?: number;
}

export interface BatchTicketItem {
  ticket_id: string;
  channel: Channel;
  subject?: string;
  text: string;
}

export interface BatchRequest {
  tickets: BatchTicketItem[];
}

export interface BatchPredictionItem extends PredictResponse {
  ticket_id: string;
}

export interface BatchResponse {
  predictions: BatchPredictionItem[];
  meta?: {
    count: number;
    model_version: string;
    processing_time_ms: number;
  };
}

export interface BatchJobStatus {
  job_id: string;
  status: JobStatus;
  total: number;
  processed: number;
  created_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  expires_at?: string | null;
  model_version?: string;
  error?: {
    code: string;
    message: string;
  } | null;
}

export interface BatchJobResults {
  job_id: string;
  status: 'succeeded';
  total: number;
  offset: number;
  limit: number;
  next_offset: number | null;
  model_version: string;
  predictions: BatchPredictionItem[];
}

export interface HealthResponse {
  status: 'ok' | 'loading';
  model_version: string | null;
  model_loaded?: boolean;
  device?: string;
}

export interface ErrorDetail {
  index?: number;
  field: string;
  issue: string;
}

export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: ErrorDetail[];
  };
}
