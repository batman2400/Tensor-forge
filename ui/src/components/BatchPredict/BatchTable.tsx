import React, { useMemo, useState } from 'react';
import { BatchPredictionItem } from '../../api/types';
import { formatCategory } from '../../lib/categories';
import { formatConfidence } from '../../lib/format';
import './BatchTable.css';

export interface BatchTableRow extends BatchPredictionItem {
  i: number;
  text: string;
}

interface BatchTableProps {
  rows: BatchTableRow[];
  urgentOnly: boolean;
}

type SortKey = 'i' | 'text' | 'category' | 'team' | 'secondary_category' | 'is_urgent' | 'confidence';

export const BatchTable: React.FC<BatchTableProps> = ({ rows, urgentOnly }) => {
  const [sortKey, setSortKey] = useState<SortKey>('i');
  const [sortDir, setSortDir] = useState<1 | -1>(1);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === 1 ? -1 : 1));
    } else {
      setSortKey(key);
      setSortDir(1);
    }
  };

  const displayRows = useMemo(() => {
    let filtered = rows;
    if (urgentOnly) {
      filtered = filtered.filter((r) => r.is_urgent);
    }

    return [...filtered].sort((a, b) => {
      let valA: unknown = a[sortKey];
      let valB: unknown = b[sortKey];

      if (valA === null || valA === undefined) valA = '';
      if (valB === null || valB === undefined) valB = '';

      if (typeof valA === 'string' && typeof valB === 'string') {
        return valA.localeCompare(valB) * sortDir;
      }
      if (typeof valA === 'boolean' && typeof valB === 'boolean') {
        return ((valA ? 1 : 0) - (valB ? 1 : 0)) * sortDir;
      }
      if (typeof valA === 'number' && typeof valB === 'number') {
        return (valA - valB) * sortDir;
      }
      return 0;
    });
  }, [rows, urgentOnly, sortKey, sortDir]);

  const renderSortArrow = (key: SortKey) => {
    if (sortKey !== key) return null;
    return <span className="sort-arrow">{sortDir === 1 ? '▲' : '▼'}</span>;
  };

  return (
    <div className="tw" tabIndex={0} aria-label="Batch predictions table">
      <table>
        <thead>
          <tr>
            <th onClick={() => handleSort('i')} style={{ width: '40px' }}>
              # {renderSortArrow('i')}
            </th>
            <th onClick={() => handleSort('text')}>
              Ticket {renderSortArrow('text')}
            </th>
            <th onClick={() => handleSort('category')}>
              Category {renderSortArrow('category')}
            </th>
            <th onClick={() => handleSort('team')}>
              Team {renderSortArrow('team')}
            </th>
            <th onClick={() => handleSort('secondary_category')}>
              Secondary {renderSortArrow('secondary_category')}
            </th>
            <th onClick={() => handleSort('is_urgent')}>
              Urgent {renderSortArrow('is_urgent')}
            </th>
            <th onClick={() => handleSort('confidence')}>
              Conf. {renderSortArrow('confidence')}
            </th>
          </tr>
        </thead>
        <tbody>
          {displayRows.length > 0 ? (
            displayRows.map((r) => (
              <tr key={r.ticket_id || r.i}>
                <td className="m">{r.i}</td>
                <td className="t">{r.text}</td>
                <td className="m">{formatCategory(r.category)}</td>
                <td>{r.team}</td>
                <td className="m">
                  {r.secondary_category ? formatCategory(r.secondary_category) : '—'}
                </td>
                <td>{r.is_urgent ? <span className="urg">URGENT</span> : ''}</td>
                <td className="m">{formatConfidence(r.confidence)}</td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={7} className="table-empty-row">
                No tickets to show.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};
