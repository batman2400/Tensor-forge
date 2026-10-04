import { Category, Team } from '../api/types';

export interface CategoryInfo {
  id: Category;
  label: string;
  team: Team;
}

export const CATEGORIES: Record<Category, CategoryInfo> = {
  payment_refund: {
    id: 'payment_refund',
    label: 'Payment & Refund',
    team: 'Payments & Refunds',
  },
  ride_trip_issue: {
    id: 'ride_trip_issue',
    label: 'Ride / Trip Issue',
    team: 'Ride Operations',
  },
  lost_item: {
    id: 'lost_item',
    label: 'Lost Item',
    team: 'Lost & Found',
  },
  order_missing_wrong: {
    id: 'order_missing_wrong',
    label: 'Order Missing / Wrong',
    team: 'Food Operations',
  },
  delivery_delay: {
    id: 'delivery_delay',
    label: 'Delivery Delay',
    team: 'Delivery Operations',
  },
  food_quality: {
    id: 'food_quality',
    label: 'Food Quality',
    team: 'Restaurant Quality',
  },
  account_promo: {
    id: 'account_promo',
    label: 'Account & Promo',
    team: 'Account Services',
  },
  safety_conduct: {
    id: 'safety_conduct',
    label: 'Safety & Conduct',
    team: 'Trust & Safety',
  },
  app_technical: {
    id: 'app_technical',
    label: 'App Technical',
    team: 'Tech Support',
  },
  general_inquiry: {
    id: 'general_inquiry',
    label: 'General Inquiry',
    team: 'Front-line Support',
  },
  spam_irrelevant: {
    id: 'spam_irrelevant',
    label: 'Spam / Irrelevant',
    team: 'Auto-close / Spam Filter',
  },
};

export function getCategoryTeam(category: Category): Team {
  return CATEGORIES[category]?.team || 'Front-line Support';
}

export function formatCategory(cat: string | null | undefined): string {
  if (!cat) return 'none';
  return cat.replace(/_/g, ' ');
}
