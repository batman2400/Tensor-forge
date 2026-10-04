/**
 * File generation and download utilities (CSV and JSON)
 */

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function downloadJSON(data: unknown, filename: string): void {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8;' });
  downloadBlob(blob, filename);
}

export function downloadCSV(
  rows: Array<Record<string, unknown>>,
  filename: string,
  headers?: string[]
): void {
  if (!rows || rows.length === 0) {
    const blob = new Blob([''], { type: 'text/csv;charset=utf-8;' });
    downloadBlob(blob, filename);
    return;
  }

  const keys = headers || Object.keys(rows[0]);
  const escapeCsv = (val: unknown): string => {
    if (val === null || val === undefined) return '';
    const str = String(val);
    if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const headerLine = keys.map(escapeCsv).join(',');
  const rowLines = rows.map((row) => keys.map((k) => escapeCsv(row[k])).join(','));
  const csvContent = [headerLine, ...rowLines].join('\r\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  downloadBlob(blob, filename);
}
