import React from 'react';
import { Zap } from 'lucide-react';
import { IChatMessage } from '@/api/chat-messages/types';
import { useTranslation } from '@/i18n';
import { formatMessageDate } from './utils';
import { Loading } from './Loading';

interface SystemMessageProps {
  message: IChatMessage;
  testId?: string;
}

// Currently we assume system message is used only for summary, in future that should change to utilize components prop or whatever ag-ui defines.
export const SystemMessage: React.FC<SystemMessageProps> = ({ message, testId }) => {
  const { t } = useTranslation();

  return (
    <div
      className="mr-2 mb-2.5 flex flex-col items-start justify-start gap-3 rounded bg-card p-3"
      data-testid={testId}
      key={message.id}
    >
      <div className="inline-flex items-center justify-between gap-1 self-stretch">
        <div className="flex h-6 shrink grow basis-0 items-center justify-start gap-2">
          <div className="flex size-6 items-center justify-center rounded-full bg-blue-100">
            <Zap className="size-4 text-blue-600" />
          </div>
          <div className="mn-label-large">{t('Summarization')}</div>
          {message.created_at && (
            <div className="body-secondary">{formatMessageDate(message.created_at)}</div>
          )}
        </div>
      </div>
      <div className="self-stretch body">
        {message.in_progress ? (
          <Loading
            statusText={t("Condensing conversation history to stay within model's context limits")}
          />
        ) : message.error ? (
          <div className="body text-destructive">{message.error}</div>
        ) : (
          <div>
            <div className="mb-2 body">
              {t(
                "Previous messages have been condensed to stay within model's context limits. Your full chat history is preserved."
              )}
            </div>
            <div className="body-secondary whitespace-pre-wrap">{message.content}</div>
          </div>
        )}
      </div>
    </div>
  );
};
