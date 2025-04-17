import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faExclamationTriangle } from "@fortawesome/free-solid-svg-icons/faExclamationTriangle";
import { CollapsiblePanel } from "./CollapsiblePanel";
import { ICodeExecutionError } from "@/api-state/chat-messages/types";
// @ts-expect-error ???
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
// @ts-expect-error ???
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

interface ErrorPanelProps {
  attempts?: number | null;
  errors?: ICodeExecutionError[];
  componentType?: string;
}

export const ErrorPanel: React.FC<ErrorPanelProps> = ({
  attempts,
  errors,
  componentType = "Component",
}) => {
  if (!errors) return null;

  return (
    <>
      {attempts && (
        <h2 className="mb-2">
          Failed to generate valid code after {attempts} attempts
        </h2>
      )}
      {errors.map((e) => {
        const { code, exception_str, stderr, stdout, traceback_str } = e;
        const hasDetails = !!(code || stderr || stdout || traceback_str);

        return (
          <div className="my-4 w-full">
            <CollapsiblePanel
              header={
                <div className="flex items-center text-destructive">
                  <FontAwesomeIcon
                    icon={faExclamationTriangle}
                    className="mr-2 flex-shrink-0"
                  />
                  <span className="font-semibold truncate">
                    {componentType} Error:{" "}
                    {exception_str || "An error occurred during execution"}
                  </span>
                </div>
              }
            >
              <div className="space-y-4">
                {code && (
                  <div>
                    <h4 className="font-semibold mb-2">
                      Code that caused the error:
                    </h4>
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
                    <h4 className="font-semibold mb-2">Traceback:</h4>
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
                    <h4 className="font-semibold mb-2">Standard Output:</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto">
                      <pre className="p-2 rounded whitespace-pre">{stdout}</pre>
                    </div>
                  </div>
                )}

                {stderr && (
                  <div>
                    <h4 className="font-semibold mb-2">Standard Error:</h4>
                    <div className="max-h-[300px] overflow-x-auto overflow-y-auto max-w-full">
                      <pre className="p-2 rounded whitespace-pre text-destructive">
                        {stderr}
                      </pre>
                    </div>
                  </div>
                )}

                {!hasDetails && (
                  <p className="italic">
                    No additional error details available.
                  </p>
                )}
              </div>
            </CollapsiblePanel>
          </div>
        );
      })}
    </>
  );
};
