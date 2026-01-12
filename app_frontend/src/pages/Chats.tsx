import React, { Suspense, lazy, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@radix-ui/react-separator';
import { Button } from '@/components/ui/button';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons/faTrash';
import { faFileArrowDown } from '@fortawesome/free-solid-svg-icons/faFileArrowDown';
import { useTranslation } from '@/i18n';
import {
  useDeleteChat,
  useFetchAllChats,
  useFetchAllMessages,
  useExport,
  usePollInProgressMessage,
} from '@/api/chat-messages/hooks';
import { InitialPrompt, UserPrompt, UserMessage, SystemMessage } from '@/components/chat';
import { ROUTES } from './routes';
import { Loading } from '@/components/ui-custom/loading';
import { RenameChatModal } from '@/components/RenameChatModal';
import { DataSourceToggle } from '@/components/DataSourceToggle';

import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { useMultipleDatasetMetadata } from '@/api/cleansed-datasets/hooks';
import { DATA_SOURCES, EXTERNAL_DATA_STORE_PREFIX } from '@/constants/dataSources';
import { Loading as ChatLoading } from '@/components/chat/Loading';
import { ConfirmDialog } from '@/components/ui-custom/confirm-dialog';
import { getChatMessageKey } from '@/components/ui-custom/prompt-input';

// Lazy load ResponseMessage for better performance
const ResponseMessage = lazy(() =>
  import('../components/chat/ResponseMessage').then(module => ({
    default: module.ResponseMessage,
  }))
);

const ComponentLoading = () => {
  return (
    <div className="p-3 bg-card rounded flex mb-2.5 mr-2">
      <ChatLoading />
    </div>
  );
};

export const Chats: React.FC = () => {
  const { t } = useTranslation();
  const { chatId } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  // API data hooks
  const { data: chats } = useFetchAllChats();
  // Find the active chat based on chatId param
  const activeChat = chats ? chats.find(chat => chat.id === chatId) : undefined;

  const { data: messages = [], isLoading: messagesLoading } = useFetchAllMessages({ chatId });
  const hasInProgressMessages = messages.some(m => m.in_progress);
  const hasFailedMessages = messages.some(m => m.error);
  usePollInProgressMessage({ chatId });

  const { mutate: deleteChat, isPending: isDeleting } = useDeleteChat({
    onSuccess: () => {
      const otherChats = chats?.filter(chat => chat.id !== activeChat?.id) || [];
      localStorage.removeItem(getChatMessageKey(activeChat?.id as string));
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
  const { exportChat, isLoading: isExporting } = useExport();
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
      } else if (data_source === DATA_SOURCES.REMOTE_CATALOG) {
        dataSourcesSet.add(DATA_SOURCES.REMOTE_CATALOG);
      } else if (data_source.startsWith(EXTERNAL_DATA_STORE_PREFIX)) {
        dataSourcesSet.add(data_source);
      }
    });

    return {
      // Users can only select data sources that are present in the metadata
      allowedDataSources: Array.from(dataSourcesSet),
      hasMixedSources: dataSourcesSet.size > 1,
    };
  }, [multipleMetadata]);

  const exportButtonTooltip = hasFailedMessages
    ? t('Cannot export chat with errors')
    : isExporting
      ? t('Exporting...')
      : hasInProgressMessages
        ? t('Wait for agent to finish responding')
        : t('Export chat');

  const isExportButtonDisabled = isExporting || hasInProgressMessages || hasFailedMessages;

  // Handler for deleting the current chat
  const handleDeleteChat = () => {
    if (activeChat?.id) {
      deleteChat({ chatId: activeChat.id });
    }
  };

  // Render the header with chat title and actions
  const renderChatHeader = () => {
    if (!activeChat) return null;

    return (
      <>
        <h2 className="heading-04 flex-1">
          <strong>{activeChat.name || t('New Chat')}</strong>
          <RenameChatModal chatId={activeChat.id} currentName={activeChat.name} />
        </h2>
        <div>
          {hasMixedSources && (
            <DataSourceToggle
              multipleMetadata={multipleMetadata}
              allowedDataSources={allowedDataSources}
            />
          )}
        </div>
        <Button
          variant="ghost"
          onClick={() => exportChat({ chatId: activeChat.id })}
          disabled={isExportButtonDisabled}
          title={exportButtonTooltip}
          testId="export-chat-button"
        >
          <FontAwesomeIcon icon={faFileArrowDown} />
          <span className="ml-2">{isExporting ? t('Exporting...') : t('Export chat')}</span>
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
      {messagesLoading && !messages?.length ? (
        <div className="flex items-center justify-center h-[calc(100vh-200px)]">
          <Loading />
        </div>
      ) : !chatId || messages?.length === 0 ? (
        <InitialPrompt
          key={`initial-${chatId || 'new'}`}
          allowedDataSources={allowedDataSources}
          chatId={activeChat?.id}
          activeChat={activeChat}
          testId="initial-prompt"
        />
      ) : (
        <>
          <ScrollArea className="flex flex-1 flex-col overflow-y-hidden pr-2 pb-4">
            {activeChat?.id &&
              messages?.map(message => (
                <div key={message.id} className="flex flex-col w-full">
                  {message.role === 'user' && (
                    <UserMessage
                      message={message}
                      chatId={activeChat.id}
                      messages={messages}
                      testId={`user-message-${message.id}`}
                    />
                  )}
                  {message.role === 'assistant' && (
                    // Suspense is needed because of lazy-loading
                    <Suspense fallback={<ComponentLoading />}>
                      <ResponseMessage
                        message={message}
                        chatId={activeChat.id}
                        messages={messages}
                        hasInProgressMessages={hasInProgressMessages}
                        testId={`response-message-${message.id}`}
                      />
                    </Suspense>
                  )}
                  {message.role === 'system' && (
                    <SystemMessage message={message} testId={`system-message-${message.id}`} />
                  )}
                </div>
              ))}
          </ScrollArea>
          <div className="flex w-full justify-center">
            <UserPrompt
              key={`user-${chatId || 'new'}`}
              allowedDataSources={allowedDataSources}
              chatId={activeChat?.id}
              activeChat={activeChat}
              hasInProgressMessages={hasInProgressMessages}
              testId="user-prompt"
            />
          </div>
        </>
      )}
    </div>
  );
};
