export const messageKeys = {
  all: ['messages', 'chats'],
  chats: ['chats'],
  messages: (chatId?: string) => ['messages', ...(chatId ? [chatId] : [])],
  singleMessage: (chatId?: string, messageId?: string) => ['message', chatId, messageId],
};
