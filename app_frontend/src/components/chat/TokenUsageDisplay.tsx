import React from 'react';
import { ITokenUsageInfo } from '@/api/chat-messages/types';

interface TokenUsageDisplayProps {
  usage: ITokenUsageInfo;
}

export const TokenUsageDisplay: React.FC<TokenUsageDisplayProps> = ({ usage }) => {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">
        {usage.total_tokens.toLocaleString()} total
      </span>
      <span className="text-xs text-muted-foreground">•</span>
      <span className="text-xs text-muted-foreground">
        {usage.prompt_tokens.toLocaleString()} input
      </span>
      <span className="text-xs text-muted-foreground">•</span>
      <span className="text-xs text-muted-foreground">
        {usage.completion_tokens.toLocaleString()} output
      </span>
      <span className="text-xs text-muted-foreground">•</span>
      <span className="text-xs text-muted-foreground">
        {usage.call_count.toLocaleString()} llm calls
      </span>
      <span className="text-xs text-muted-foreground">•</span>
      <span className="text-xs text-muted-foreground mr-2">{usage.model}</span>
    </div>
  );
};

export default TokenUsageDisplay;
