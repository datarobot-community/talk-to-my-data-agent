import React, { useEffect, useRef } from 'react';
import { MessageHeader } from './MessageHeader';
import { formatMessageDate } from './utils';
import { useDeleteMessage } from '@/api/chat-messages/hooks';

interface UserMessageProps {
    id?: string;
    date?: string;
    timestamp?: string;
    message?: string;
    chatId?: string;
    responseId?: string;
}

export const UserMessage: React.FC<UserMessageProps> = ({
    id,
    date,
    timestamp,
    message,
    chatId,
    responseId,
}) => {
    const ref = useRef<HTMLDivElement>(null);
    const { mutate: deleteMessage } = useDeleteMessage();

    useEffect(() => {
        ref.current?.scrollIntoView(false);
    });

    // Use the formatted timestamp if available, otherwise fallback to date prop or default
    const displayDate = timestamp ? formatMessageDate(timestamp) : date || '';

    const handleDelete = () => {
        if (id) {
            deleteMessage({
                messageId: id,
                chatId: chatId,
            });
        }
        if (responseId) {
            deleteMessage({
                messageId: responseId,
                chatId: chatId,
            });
        }
    };

    return (
        <div
            className="p-3 bg-card rounded flex-col justify-start items-start gap-3 flex mb-2.5 mr-2"
            ref={ref}
        >
            <MessageHeader name={'You'} date={displayDate} onDelete={handleDelete} />
            <div className="self-stretch text-sm font-normal leading-tight">{message}</div>
        </div>
    );
};
