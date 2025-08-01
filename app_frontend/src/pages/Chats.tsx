import React, { Suspense, lazy, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@radix-ui/react-separator';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { faFileArrowDown } from '@fortawesome/free-solid-svg-icons/faFileArrowDown';
import { useTranslation } from '@/i18n';
import { useFetchAllMessages, useDeleteChat, useFetchAllChats } from '@/api/chat-messages/hooks';
import { InitialPrompt, UserMessage, UserPrompt } from '@/components/chat';
import { ROUTES } from './routes';
import { Loading } from '@/components/ui-custom/loading';
import { RenameChatModal } from '@/components/RenameChatModal';
import { DataSourceToggle } from '@/components/DataSourceToggle';
import { IChatMessage } from '@/api/chat-messages/types';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { useMultipleDatasetMetadata } from '@/api/cleansed-datasets/hooks';
import { DATA_SOURCES } from '@/constants/dataSources';
import { handleDownload } from '@/api/chat-messages/api-requests.ts';
import { ConfirmDialog } from '@/components/ui-custom/confirm-dialog';

// Lazy load ResponseMessage for better performance
const ResponseMessage = lazy(() =>
  import('../components/chat/ResponseMessage').then(module => ({
    default: module.ResponseMessage,
  }))
);

const ComponentLoading = () => {
  const { t } = useTranslation();
  return <div className="p-4 text-sm">{t('Loading component...')}</div>;
};

const ChatMessageItem = ({
  message,
  messages,
  chatId,
  index,
}: {
  message: IChatMessage;
  messages: IChatMessage[];
  chatId?: string;
  index: number;
}) => {
  const responseId = messages[index + 1]?.id;

  return (
    <div className="table table-fixed w-full">
      {message.role === 'user' ? (
        <>
          <UserMessage
            id={message.id}
            responseId={responseId}
            message={message.content}
            timestamp={message.created_at}
            chatId={chatId}
          />
          {message.in_progress && (
            <Suspense fallback={<ComponentLoading />}>
              <ResponseMessage message={message} isLoading={true} chatId={chatId} />
            </Suspense>
          )}
        </>
      ) : (
        <Suspense fallback={<ComponentLoading />}>
          <ResponseMessage message={message} isLoading={false} chatId={chatId} />
        </Suspense>
      )}
    </div>
  );
};

export const Chats: React.FC = () => {
  const { t } = useTranslation();
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  // API data hooks
  const { data: messages = [], status: messagesStatus } = useFetchAllMessages({
    chatId,
  });
  const { data: chats } = useFetchAllChats();
  // Find the active chat based on chatId param
  const activeChat = chats ? chats.find(chat => chat.id === chatId) : undefined;
  const { mutate: deleteChat, isPending: isDeleting } = useDeleteChat({
    onSuccess: () => {
      const otherChats = chats?.filter(chat => chat.id !== activeChat?.id) || [];

      setIsDeleteDialogOpen(false);
      if (otherChats.length > 0) {
        // Sort by creation date descending (newest first) to get the most recent chat
        const sortedChats = [...otherChats].sort((a, b) => {
          const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
          const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
          return dateB - dateA; // Descending order (newest first)
        });

        // Navigate to the most recent chat
        navigate(`/chats/${sortedChats[0].id}`);
      } else {
        // If no other chats, go to the main chats page
        navigate(ROUTES.CHATS);
      }
    },
  });
  const isProcessing = messages?.some(message => message.in_progress);
  const { data: dictionaries } = useGeneratedDictionaries();
  const { data: multipleMetadata } = useMultipleDatasetMetadata(
    dictionaries?.map(d => d.name) || []
  );

  const { hasMixedSources, allowedDataSources } = useMemo(() => {
    if (!multipleMetadata) return { allowedDataSources: [], hasMixedSources: false };

    const dataSourcesSet = new Set<string>();

    multipleMetadata.forEach(({ metadata }) => {
      const { data_source } = metadata;

      if (data_source === DATA_SOURCES.FILE || data_source === DATA_SOURCES.CATALOG) {
        dataSourcesSet.add(DATA_SOURCES.FILE);
      } else if (data_source === DATA_SOURCES.DATABASE) {
        dataSourcesSet.add(DATA_SOURCES.DATABASE);
      }
    });

    return {
      // Users can only select data sources that are present in the metadata
      allowedDataSources: Array.from(dataSourcesSet),
      hasMixedSources:
        dataSourcesSet.has(DATA_SOURCES.FILE) && dataSourcesSet.has(DATA_SOURCES.DATABASE),
    };
  }, [multipleMetadata]);

  // Handler for deleting the current chat
  const handleDeleteChat = () => {
    if (activeChat?.id) {
      deleteChat({ chatId: activeChat.id });
    }
  };

  // Render chat messages
  const renderMessages = () => {
    if (!messages || messages.length === 0) {
      return (
        <Suspense fallback={<ComponentLoading />}>
          <InitialPrompt allowedDataSources={allowedDataSources} chatId={activeChat?.id} />
        </Suspense>
      );
    }

    return (
      <>
        <ScrollArea className="flex flex-1 flex-col overflow-y-hidden pr-2 pb-4">
          {messages?.map((message, index, array) => (
            <ChatMessageItem
              key={index}
              message={message}
              messages={array}
              chatId={activeChat?.id}
              index={index}
            />
          ))}
        </ScrollArea>
        <div className="flex w-full justify-center">
          <Suspense fallback={<ComponentLoading />}>
            <UserPrompt
              allowedDataSources={allowedDataSources}
              chatId={activeChat?.id}
              isProcessing={isProcessing}
            />
          </Suspense>
        </div>
      </>
    );
  };

  // Render the header with chat title and actions
  const renderChatHeader = () => {
    if (!activeChat) return null;

    return (
      <>
        <ConfirmDialog
          open={isDeleteDialogOpen}
          isLoading={isDeleting}
          onOpenChange={setIsDeleteDialogOpen}
          title={t('Delete chat')}
          confirmText={t('Delete')}
          cancelText={t('Cancel')}
          description={t('Are you sure you want to delete this chat?')}
          onConfirm={handleDeleteChat}
          variant="destructive"
        />
        <h2 className="text-xl flex-1">
          <strong>{activeChat.name || t('New Chat')}</strong>
          <RenameChatModal chatId={activeChat.id} currentName={activeChat.name} />
        </h2>
        <div>{hasMixedSources && <DataSourceToggle multipleMetadata={multipleMetadata} />}</div>
        <Button variant="ghost" onClick={() => handleDownload(activeChat.id)}>
          <FontAwesomeIcon icon={faFileArrowDown} />
          <span className="ml-2">{t('Export chat')}</span>
        </Button>
        <Button
          variant="ghost"
          onClick={() => setIsDeleteDialogOpen(true)}
          testId="delete-all-chats-button"
        >
          <FontAwesomeIcon icon={faTrash} />
          <span className="ml-2">{t('Delete chat')}</span>
        </Button>
      </>
    );
  };

  return (
    <div className="p-6 h-full flex flex-col">
      <div className="flex justify-between items-center gap-2 h-9">{renderChatHeader()}</div>
      <Separator className="my-4 border-t" />

      {messagesStatus === 'pending' && !messages?.length ? (
        <div className="flex items-center justify-center h-[calc(100vh-200px)]">
          <Loading />
        </div>
      ) : (
        renderMessages()
      )}
    </div>
  );
};
