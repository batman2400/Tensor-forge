import { Category, PredictRequest, PredictResponse, Team } from '../src/api/types';
import { CATEGORIES } from '../src/lib/categories';

interface PatternRule {
  pattern: RegExp;
  category: Category;
  secondary?: Category;
}

const RULES: PatternRule[] = [
  { pattern: /refund|charged|double charge|duplicate|overcharge|billing|credit card/i, category: 'payment_refund', secondary: 'account_promo' },
  { pattern: /lost|left behind|forgot|seat|wallet|keys|backpack|bag/i, category: 'lost_item', secondary: 'ride_trip_issue' },
  { pattern: /driver|route|pickup|dropoff|detour|car|vehicle|navigation/i, category: 'ride_trip_issue', secondary: 'safety_conduct' },
  { pattern: /missing item|wrong item|forgot burger|missing drink|sauce missing/i, category: 'order_missing_wrong', secondary: 'food_quality' },
  { pattern: /cold|spoiled|undercooked|raw|hair|taste|spilled/i, category: 'food_quality', secondary: 'order_missing_wrong' },
  { pattern: /late|delay|waiting|on the way|where is my food|slow delivery/i, category: 'delivery_delay', secondary: 'order_missing_wrong' },
  { pattern: /unsafe|reckless|harass|accident|threat|emergency|speeding|red light|police/i, category: 'safety_conduct', secondary: 'ride_trip_issue' },
  { pattern: /password|login|otp|locked|promo|discount|voucher|coupon|points/i, category: 'account_promo', secondary: 'general_inquiry' },
  { pattern: /crash|bug|freeze|black screen|error code|reinstall|update/i, category: 'app_technical', secondary: 'general_inquiry' },
  { pattern: /spam|casino|lottery|crypto|subscribe|unsubscribe|viagra|click here/i, category: 'spam_irrelevant' },
];

const URGENT_REGEX = /unsafe|accident|harass|threat|emergency|stranded|charged twice|fraud|police|injury/i;

export function mockPredict(req: PredictRequest): PredictResponse {
  const fullText = `${req.subject || ''} ${req.text}`.trim();
  let matchedCategory: Category = 'general_inquiry';
  let matchedSecondary: Category | null = null;

  for (const rule of RULES) {
    if (rule.pattern.test(fullText)) {
      matchedCategory = rule.category;
      matchedSecondary = rule.secondary || null;
      break;
    }
  }

  const isUrgent = URGENT_REGEX.test(fullText);

  // Calibrated pseudo-random confidence based on text hash
  let hash = 0;
  for (let i = 0; i < fullText.length; i++) {
    hash = (hash * 31 + fullText.charCodeAt(i)) & 0xffffffff;
  }
  const normalized = Math.abs(hash % 1000) / 1000;
  const confidence = +(0.75 + normalized * 0.23).toFixed(2);
  const latency = 12 + Math.abs(hash % 25);

  const team: Team = CATEGORIES[matchedCategory]?.team || 'Front-line Support';

  return {
    ticket_id: req.ticket_id,
    category: matchedCategory,
    secondary_category: matchedSecondary,
    team,
    is_urgent: isUrgent,
    confidence,
    model_version: 'v2.0.0-phase2-prod',
    needs_human_review: confidence < 0.8,
    latency_ms: latency,
  };
}
