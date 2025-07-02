export const messageKeys = {
    all: ['messages', 'chats'],
    chats: ['chats'],
    messages: (chatId?: string) => ['messages', ...(chatId ? [chatId] : [])],
};
