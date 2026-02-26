import React from 'react';
import { TriangleAlert } from 'lucide-react';
import { CollapsiblePanel } from './CollapsiblePanel';
import { ICodeExecutionError } from '@/api/chat-messages/types';
import { useTranslation } from '@/i18n';
// @ts-expect-error ???
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
// @ts-expect-error ???
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ErrorPanelProps {
  attempts?: number | null;
  errors?: ICodeExecutionError[];
  componentType?: string;
}

export const ErrorPanel: React.FC<ErrorPanelProps> = ({
  attempts,
  errors,
  componentType = 'Component',
}) => {
  const { t } = useTranslation();
  if (!errors || errors.length === 0) return null;
  return (
    <>
      {attempts && (
        <h2 className="mb-2">
          {t('Failed to generate valid code after {{attempts}} attempts', { attempts })}
        </h2>
      )}
      {errors.map(e => {
        const { code, exception_str, stderr, stdout, traceback_str } = e;
        const hasDetails = !!(code || stderr || stdout || traceback_str);
        const error = `${componentType} Error: ${
          exception_str || t('An error occurred during execution')
        }`;

        return (
          <div className="my-4 w-full">
            <CollapsiblePanel
              header={
                <div className="flex items-center text-destructive">
                  <TriangleAlert className="mr-2 size-4 flex-shrink-0" />
                  <span className="font-semibold">{error}</span>
                </div>
              }
            >
              <div className="space-y-4">
                {code && (
                  <div>
                    <h4 className="mb-2 font-semibold">{t('Code that caused the error:')}</h4>
                    <div className="max-h-[500px] overflow-x-auto overflow-y-auto">
                      <SyntaxHighlighter
                        language="python"
                        style={oneDark}
                        className="rounded"
                        customStyle={{ margin: 0 }}
                        wrapLongLines={false}
                        showLineNumbers={true}
                      >
                        {code}
                      </SyntaxHighlighter>
                    </div>
                  </div>
                )}

                {traceback_str && (
                  <div>
                    <h4 className="mb-2 font-semibold">{t('Traceback:')}</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto">
                      <SyntaxHighlighter
                        language="python"
                        style={oneDark}
                        className="rounded"
                        customStyle={{ margin: 0 }}
                        wrapLongLines={false}
                      >
                        {traceback_str}
                      </SyntaxHighlighter>
                    </div>
                  </div>
                )}

                {stdout && (
                  <div>
                    <h4 className="mb-2 font-semibold">{t('Standard Output:')}</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto">
                      <pre className="rounded p-2 whitespace-pre">{stdout}</pre>
                    </div>
                  </div>
                )}

                {stderr && (
                  <div>
                    <h4 className="mb-2 font-semibold">{t('Standard Error:')}</h4>
                    <div className="max-h-[300px] max-w-full overflow-x-auto overflow-y-auto">
                      <pre className="rounded p-2 whitespace-pre text-destructive">{stderr}</pre>
                    </div>
                  </div>
                )}

                {!hasDetails && (
                  <p className="italic">{t('No additional error details available.')}</p>
                )}
              </div>
            </CollapsiblePanel>
          </div>
        );
      })}
    </>
  );
};
