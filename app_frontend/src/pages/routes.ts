export const ROUTES = {
  DATA: '/data',
  CHATS: '/chats',
  CHAT_WITH_ID: '/chats/:chatId',
  DATA_WITH_ID: '/data/:dataId',
};

export const generateChatRoute = (chatId?: string) => {
  if (!chatId) return ROUTES.CHATS;
  return `/chats/${chatId}`;
};

export const generateDataRoute = (dataId?: string) => {
  if (!dataId) return ROUTES.DATA;
  return `/data/${dataId}`;
};
