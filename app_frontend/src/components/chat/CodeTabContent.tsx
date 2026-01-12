import React from 'react';
import { CollapsiblePanel } from './CollapsiblePanel';
import { CleansedDataTable } from '../data/CleansedDataTable';
import { useTranslation } from '@/i18n';
// @ts-expect-error ???
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// @ts-expect-error ???
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './MarkdownContent.css';
import { Button } from '@/components/ui/button';
import { useDownloadDataset } from '@/api/datasets/hooks';
import { useAppState } from '@/state';
import { Download, Loader2 } from 'lucide-react';
import { CopyToClipboardButton } from '@/components/ui-custom/copy-to-clipboard-button';
import { useTheme } from '@/theme/theme-provider';

interface CodeTabContentProps {
  code?: string | null;
  datasetId?: string | null;
}

export const CodeTabContent: React.FC<CodeTabContentProps> = ({ code, datasetId }) => {
  const { t } = useTranslation();
  const { mutate: downloadDataset, isPending: isDownloadingDataset } = useDownloadDataset();
  const { includeCsvBom } = useAppState();
  const { theme } = useTheme();

  const downloadTooltip = isDownloadingDataset ? t('Downloading...') : t('Download dataset as CSV');

  return (
    <div className="flex flex-col gap-2.5 min-w-0">
      {datasetId && (
        <CollapsiblePanel header={t('Analysis Dataset')}>
          <div className="flex flex-col gap-2">
            <div className="flex justify-end">
              <Button
                variant="ghost"
                onClick={() => downloadDataset({ datasetId, includeBom: includeCsvBom })}
                disabled={isDownloadingDataset}
                title={downloadTooltip}
              >
                {isDownloadingDataset ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                {t('Download dataset')}
              </Button>
            </div>
            <CleansedDataTable datasetId={datasetId} maxHeight="max-h-[400px]" />
          </div>
        </CollapsiblePanel>
      )}
      {code && (
        <CollapsiblePanel header={t('Code')}>
          <div className="flex flex-col gap-2">
            <div className="flex justify-end">
              <CopyToClipboardButton content={code} label={t('Copy snippet to clipboard')} />
            </div>
            <div className="markdown-content">
              <SyntaxHighlighter
                language="python"
                style={theme === 'dark' ? oneDark : oneLight}
                customStyle={{
                  margin: 0,
                  borderRadius: '4px',
                }}
                wrapLongLines={true}
                // Note: showLineNumbers={true} breaks text selection for copy - react-syntax-highlighter known issue
              >
                {code}
              </SyntaxHighlighter>
            </div>
          </div>
        </CollapsiblePanel>
      )}
    </div>
  );
};
