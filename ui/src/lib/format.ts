/**
 * Formatting utility functions for Dispatch Desk UI
 */

export function formatConfidence(conf: number): string {
  if (typeof conf !== 'number' || isNaN(conf)) return '0%';
  return `${Math.round(conf * 100)}%`;
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}

export function formatLatency(ms?: number): string {
  if (typeof ms !== 'number') return '— ms';
  return `${Math.round(ms)} ms`;
}

export function truncate(str: string, maxLen = 60): string {
  if (!str) return '';
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + '…';
}
