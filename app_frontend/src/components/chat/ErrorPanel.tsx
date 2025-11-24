import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExclamationTriangle } from '@fortawesome/free-solid-svg-icons/faExclamationTriangle';
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
                  <FontAwesomeIcon icon={faExclamationTriangle} className="mr-2 flex-shrink-0" />
                  <span className="font-semibold">{error}</span>
                </div>
              }
            >
              <div className="space-y-4">
                {code && (
                  <div>
                    <h4 className="font-semibold mb-2">{t('Code that caused the error:')}</h4>
                    <div className="overflow-x-auto overflow-y-auto max-h-[500px]">
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
                    <h4 className="font-semibold mb-2">{t('Traceback:')}</h4>
                    <div className="overflow-x-auto overflow-y-auto max-h-[300px]">
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
                    <h4 className="font-semibold mb-2">{t('Standard Output:')}</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto">
                      <pre className="p-2 rounded whitespace-pre">{stdout}</pre>
                    </div>
                  </div>
                )}

                {stderr && (
                  <div>
                    <h4 className="font-semibold mb-2">{t('Standard Error:')}</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto max-w-full">
                      <pre className="p-2 rounded whitespace-pre text-destructive">{stderr}</pre>
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
