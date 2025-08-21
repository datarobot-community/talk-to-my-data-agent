import { IChat, IChatMessage } from './types';
import { SidebarMenuOptionType } from '@/components/ui-custom/sidebar-menu';

export const getChatsMenu = (data: IChat[]): SidebarMenuOptionType[] => {
  const sortedChats = data?.slice().sort((a, b) => {
    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateB - dateA;
  });

  return sortedChats?.map(c => ({
    key: c.id,
    name: c.name,
  }));
};

export const getMessage = (
  messages: IChatMessage[],
  messageId: string | undefined | null
): IChatMessage | undefined => {
  if (!messageId) {
    return undefined;
  }
  return messages.find(message => message.id === messageId);
};

export const getResponseMessage = (
  messages: IChatMessage[],
  messageId: string | undefined | null
): IChatMessage | undefined => {
  if (!messageId) {
    return undefined;
  }
  const message = getMessage(messages, messageId);
  if (!message) return undefined;

  // If it's already an assistant message, return it
  if (message.role === 'assistant') return message;

  // If it's a user message, find the next assistant message
  const userIndex = messages.findIndex(msg => msg.id === messageId);
  const responseMessage = messages[userIndex + 1];
  return responseMessage?.role === 'assistant' ? responseMessage : undefined;
};
