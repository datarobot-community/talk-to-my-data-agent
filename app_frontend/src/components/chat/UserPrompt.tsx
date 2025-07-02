import { useMemo, useState } from 'react';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons/faPaperPlane';
import { usePostMessage, useFetchAllChats } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state';
import { DATA_SOURCES } from '@/constants/dataSources';

export const UserPrompt = ({
    chatId,
    allowSend,
    allowedDataSources,
}: {
    chatId?: string;
    allowSend?: boolean;
    allowedDataSources?: string[];
}) => {
    const { mutate } = usePostMessage();
    const {
        enableChartGeneration,
        enableBusinessInsights,
        dataSource: globalDataSource,
    } = useAppState();
    const { data: chats } = useFetchAllChats();
    const isDisabled = !allowedDataSources?.[0];

    const [message, setMessage] = useState('');

    // Find the active chat to get its data source setting
    const activeChat = chatId ? chats?.find(chat => chat.id === chatId) : undefined;
    const chatDataSource = useMemo(() => {
        const dataSource = activeChat?.data_source || globalDataSource;
        // User can only select from the allowed data sources
        return allowedDataSources?.includes(dataSource)
            ? dataSource
            : allowedDataSources?.[0] || DATA_SOURCES.FILE;
    }, [activeChat?.data_source, globalDataSource, allowedDataSources]);

    const sendMessage = () => {
        if (message.trim()) {
            mutate({
                message,
                chatId,
                enableChartGeneration,
                enableBusinessInsights,
                dataSource: chatDataSource,
            });
            setMessage('');
        }
    };

    return (
        <PromptInput
            icon={FontAwesomeIcon}
            iconProps={{
                icon: isDisabled ? null : faPaperPlane,
                behavior: 'append',
                onClick: sendMessage,
            }}
            placeholder={
                isDisabled
                    ? 'Please upload and process data using the sidebar before starting the chat'
                    : 'Ask another question about your datasets.'
            }
            onKeyDown={e => {
                if (e.key === 'Enter' && allowSend) {
                    sendMessage();
                }
            }}
            disabled={isDisabled}
            aria-disabled={isDisabled}
            onChange={e => setMessage(e.target.value)}
            value={message}
        />
    );
};
