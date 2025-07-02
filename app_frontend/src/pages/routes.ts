export const ROUTES = {
    DATA: '/data',
    CHATS: '/chats',
    CHAT_WITH_ID: '/chats/:chatId',
};

export const generateChatRoute = (chatId?: string) => {
    if (!chatId) return ROUTES.CHATS;
    return `/chats/${chatId}`;
};
