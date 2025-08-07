import React, { useRef, useEffect, useImperativeHandle, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons/faPaperPlane';
import { faHourglassHalf } from '@fortawesome/free-solid-svg-icons/faHourglassHalf';
import { useTranslation } from '@/i18n';

import { cn } from '~/lib/utils';
import { Button } from '@/components/ui/button';

export type SendButtonArrangement = 'prepend' | 'append';

export type PromptInputProps = Omit<
  React.TextareaHTMLAttributes<HTMLTextAreaElement>,
  'onChange' | 'value' | 'onSend'
> & {
  sendButtonArrangement?: SendButtonArrangement;
  onSend?: (message: string) => void;
  isProcessing?: boolean;
  isDisabled?: boolean;
  testId?: string;
  initialValue?: string;
};

const PromptInput = React.forwardRef<HTMLTextAreaElement, PromptInputProps>(
  (
    {
      className,
      sendButtonArrangement = 'append',
      onSend,
      isProcessing = false,
      isDisabled,
      testId = 'chat-prompt-input',
      initialValue = '',
      ...props
    },
    ref
  ) => {
    const { t } = useTranslation();
    const internalRef = useRef<HTMLTextAreaElement>(null);
    const [message, setMessage] = useState(initialValue);

    useImperativeHandle(ref, () => internalRef.current as HTMLTextAreaElement);
    useEffect(() => {
      // Auto-resize textarea based on content
      const textarea = internalRef.current;
      if (textarea) {
        textarea.style.height = 'auto'; // Reset height to find actual value

        const textHeight = textarea.scrollHeight;
        textarea.style.height = `${textHeight}px`;
        textarea.style.overflow = textHeight > 300 ? 'auto' : 'hidden';
      }
    }, [message]);

    const [isFocused, setIsFocused] = React.useState(false);
    const [isComposing, setIsComposing] = React.useState(false);

    const handleSend = () => {
      if (message.trim()) {
        onSend?.(message);
        setMessage('');
      }
    };

    const isButtonDisabled = isProcessing || isDisabled || !message.trim();

    const buttonTooltip =
      isButtonDisabled && !isProcessing
        ? t('Ask a question')
        : isButtonDisabled && isProcessing
          ? t('Processing... Waiting for response.')
          : t('Send message');

    return (
      <div
        id="prompt-input-container"
        aria-disabled={isDisabled}
        data-testid={testId}
        className={cn(
          'flex gap-2 justify-start items-center px-3 py-3 w-full min-w-3xs mr-4',
          'border-input rounded-md border shadow-xs',
          'bg-transparent placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground',
          'text-base transition-[color,box-shadow]',
          'ring-ring/10 outline-ring/50',
          isFocused &&
            'ring-4 outline-1 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
          sendButtonArrangement === 'prepend' ? 'flex-row-reverse' : 'flex-row',
          className
        )}
      >
        <textarea
          id="prompt-input-textarea"
          className={cn(
            'flex leading-5 box-content max-h-[300px] justify-center w-full resize-none overflow-hidden bg-transparent placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50',
            className
          )}
          rows={1}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          onKeyDown={event => {
            if (
              !isComposing &&
              !isProcessing &&
              !isDisabled &&
              event.key === 'Enter' &&
              !(event.shiftKey || event.altKey)
            ) {
              event.preventDefault();
              handleSend();
            }
          }}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onChange={e => setMessage(e.target.value)}
          value={message}
          ref={internalRef}
          disabled={isDisabled}
          {...props}
        />
        <Button
          variant="ghost"
          testId="send-message-button"
          disabled={isButtonDisabled}
          onClick={handleSend}
          title={buttonTooltip}
        >
          <FontAwesomeIcon icon={isProcessing ? faHourglassHalf : faPaperPlane} size="lg" />
        </Button>
      </div>
    );
  }
);
PromptInput.displayName = 'PromptInput';

export { PromptInput };
