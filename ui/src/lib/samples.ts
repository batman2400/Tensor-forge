import { Channel } from '../api/types';

export interface SampleTicket {
  label: string;
  channel: Channel;
  subject?: string;
  text: string;
}

export const SAMPLE_TICKETS: SampleTicket[] = [
  {
    label: 'Double charge',
    channel: 'email',
    subject: 'Charged twice for ride #8832',
    text: 'Hi, I took a ride this morning and my card was charged twice. Please refund the duplicate payment. Order ref is 8832.',
  },
  {
    label: 'Late delivery',
    channel: 'chat',
    text: 'My food order has been showing "on the way" for 50 minutes now and no one is picking up the phone. Where is my order?',
  },
  {
    label: 'Lost bag',
    channel: 'chat',
    text: 'I left my laptop bag in the back seat of my ride. Can you please contact the driver? Trip was around 3pm today.',
  },
  {
    label: 'Unsafe driver',
    channel: 'call_transcript',
    text: 'customer: yes hi I need to report my driver he was on his phone the entire ride and ran a red light I was terrified',
  },
  {
    label: 'App crash',
    channel: 'chat',
    text: 'The app keeps freezing whenever I try to open my order history. I have reinstalled twice and the problem is still there.',
  },
  {
    label: 'General question',
    channel: 'email',
    subject: 'Loyalty points question',
    text: 'Hello, could you explain how the loyalty points system works? I have been using the app for months but never received any points.',
  },
];
