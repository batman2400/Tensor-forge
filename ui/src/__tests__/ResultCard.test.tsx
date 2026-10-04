import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { PredictResponse } from '../api/types';
import { ResultCard } from '../components/SinglePredict/ResultCard';

describe('ResultCard', () => {
  it('renders empty desk message when result is null', () => {
    render(<ResultCard result={null} request={null} />);
    expect(screen.getByText(/Nothing on the desk yet/i)).toBeInTheDocument();
  });

  it('renders category stamp, team name, and confidence', () => {
    const mockResult: PredictResponse = {
      ticket_id: 'TF-001',
      category: 'payment_refund',
      secondary_category: 'account_promo',
      team: 'Payments & Refunds',
      is_urgent: false,
      confidence: 0.88,
      model_version: 'v2.0.0-phase2-prod',
      latency_ms: 24,
    };

    render(
      <ResultCard
        result={mockResult}
        request={{ channel: 'chat', text: 'Double charge' }}
      />
    );

    expect(screen.getByText('payment refund')).toBeInTheDocument();
    expect(screen.getByText('Payments & Refunds')).toBeInTheDocument();
    expect(screen.getByText('88%')).toBeInTheDocument();
    expect(screen.getByText('account promo')).toBeInTheDocument();
    expect(screen.queryByText('Urgent')).not.toBeInTheDocument();
  });

  it('renders URGENT stamp when is_urgent is true', () => {
    const mockResult: PredictResponse = {
      ticket_id: 'TF-002',
      category: 'safety_conduct',
      secondary_category: null,
      team: 'Trust & Safety',
      is_urgent: true,
      confidence: 0.95,
      model_version: 'v2.0.0-phase2-prod',
      latency_ms: 18,
    };

    render(
      <ResultCard
        result={mockResult}
        request={{ channel: 'call_transcript', text: 'Driver threatened customer' }}
      />
    );

    expect(screen.getByText('Urgent')).toBeInTheDocument();
  });

  it('toggles JSON / cURL panel and hides real API key', () => {
    const mockResult: PredictResponse = {
      ticket_id: 'TF-003',
      category: 'lost_item',
      secondary_category: null,
      team: 'Lost & Found',
      is_urgent: false,
      confidence: 0.84,
      model_version: 'v2.0.0-phase2-prod',
      latency_ms: 15,
    };

    render(
      <ResultCard
        result={mockResult}
        request={{ channel: 'chat', text: 'Lost phone in cab' }}
      />
    );

    const toggleBtn = screen.getByText(/view JSON \/ cURL/i);
    expect(toggleBtn).toBeInTheDocument();

    fireEvent.click(toggleBtn);
    expect(screen.getByText(/curl -X POST/i)).toBeInTheDocument();
    expect(screen.getByText(/X-API-Key: \$API_KEY/i)).toBeInTheDocument();
  });
});
