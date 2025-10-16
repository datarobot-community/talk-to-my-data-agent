import React from 'react';
import { CollapsiblePanel } from './CollapsiblePanel';
import { CleansedDataTable } from '../data/CleansedDataTable';
import { useTranslation } from '@/i18n';
// @ts-expect-error ???
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// @ts-expect-error ???
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './MarkdownContent.css';

interface CodeTabContentProps {
  code?: string | null;
  datasetId?: string | null;
}

export const CodeTabContent: React.FC<CodeTabContentProps> = ({ code, datasetId }) => {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-2.5 min-w-0">
      {datasetId && (
        <CollapsiblePanel header={t('Dataset')}>
          <CleansedDataTable datasetId={datasetId} />
        </CollapsiblePanel>
      )}
      {code && (
        <CollapsiblePanel header={t('Code')}>
          <div className="markdown-content">
            <SyntaxHighlighter
              language="python"
              style={oneDark}
              customStyle={{
                margin: 0,
                borderRadius: '4px',
              }}
              wrapLongLines={true}
              showLineNumbers={true}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        </CollapsiblePanel>
      )}
    </div>
  );
};
