import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { BatchTable, BatchTableRow } from '../components/BatchPredict/BatchTable';

describe('BatchTable', () => {
  const mockRows: BatchTableRow[] = [
    {
      i: 1,
      ticket_id: 'TF-001',
      text: 'My order is 40 min late',
      category: 'delivery_delay',
      secondary_category: null,
      team: 'Delivery Operations',
      is_urgent: false,
      confidence: 0.89,
      model_version: 'v2.0.0-phase2-prod',
    },
    {
      i: 2,
      ticket_id: 'TF-002',
      text: 'Accident during ride driver hit pole',
      category: 'safety_conduct',
      secondary_category: 'ride_trip_issue',
      team: 'Trust & Safety',
      is_urgent: true,
      confidence: 0.98,
      model_version: 'v2.0.0-phase2-prod',
    },
  ];

  it('renders all rows when urgentOnly is false', () => {
    render(<BatchTable rows={mockRows} urgentOnly={false} />);
    expect(screen.getByText('My order is 40 min late')).toBeInTheDocument();
    expect(screen.getByText('Accident during ride driver hit pole')).toBeInTheDocument();
    expect(screen.getByText('URGENT')).toBeInTheDocument();
  });

  it('filters out non-urgent tickets when urgentOnly is true', () => {
    render(<BatchTable rows={mockRows} urgentOnly={true} />);
    expect(screen.queryByText('My order is 40 min late')).not.toBeInTheDocument();
    expect(screen.getByText('Accident during ride driver hit pole')).toBeInTheDocument();
  });

  it('shows empty message when rows array is empty', () => {
    render(<BatchTable rows={[]} urgentOnly={false} />);
    expect(screen.getByText('No tickets to show.')).toBeInTheDocument();
  });

  it('sorts columns when clicking header', () => {
    render(<BatchTable rows={mockRows} urgentOnly={false} />);
    const confidenceHeader = screen.getByText(/Conf\./i);
    fireEvent.click(confidenceHeader);

    // After sort click, confidence sort arrow should appear
    expect(screen.getByText('▲')).toBeInTheDocument();
  });
});
