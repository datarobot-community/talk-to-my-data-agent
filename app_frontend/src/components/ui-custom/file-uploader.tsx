import React, { useState } from 'react';
import Dropzone, { type FileRejection } from 'react-dropzone';
import { Folder, XIcon, FileIcon } from 'lucide-react';
import { Progress } from '@/components/ui/progress';
import { useTranslation } from '@/i18n';
interface FileUploaderProps {
  maxSize?: number;
  accept?: { [key: string]: string[] };
  onFilesChange: (files: File[]) => void;
  progress: number;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  maxSize = 1024 * 1024 * 200,
  accept = { 'file/csv': ['.csv'], 'file/xlsx': ['.xlsx', '.xls'] },
  progress = 0,
  onFilesChange,
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const { t } = useTranslation();
  const onDrop = React.useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      const newFiles = acceptedFiles.map(file =>
        Object.assign(file, {
          preview: URL.createObjectURL(file),
        })
      );

      const updatedFiles = files ? [...files, ...newFiles] : newFiles;

      setFiles(updatedFiles);
      onFilesChange(updatedFiles);

      if (rejectedFiles.length > 0) {
        rejectedFiles.forEach(({ file }) => {
          console.error(`File ${file.name} was rejected`);
        });
      }
    },

    [files, onFilesChange]
  );

  function onRemove(index: number) {
    if (!files) return;
    const newFiles = files.filter((_, i) => i !== index);
    setFiles(newFiles);
    onFilesChange(newFiles);
  }

  return (
    <Dropzone onDrop={onDrop} maxSize={maxSize} accept={accept}>
      {({ getRootProps, getInputProps }) => (
        <section>
          <div
            {...getRootProps()}
            className="cu rounded-lg border border-dashed border-primary/20 p-4"
          >
            <input {...getInputProps()} aria-label={t('Upload CSV, XLSX, or XLS files')} />
            <p className="p-6 text-center">
              {t('Drag and drop from your desktop, or')} <Folder className="inline size-4" />{' '}
              <strong>{t('browse local files')}</strong>
            </p>
            <div>
              {files.map((file, index) => (
                <div
                  key={index}
                  className="h-min-[36px] flex flex-col items-start justify-start gap-2.5 pt-4"
                >
                  <div className="inline-flex items-center justify-start gap-2 self-stretch rounded border border-primary/10 bg-secondary/50">
                    <div className="inline-flex w-9 flex-col items-center justify-center gap-2 self-stretch rounded-l-[3px] bg-secondary">
                      <div className="flex size-9 flex-col items-center justify-center gap-2.5">
                        <div className="text-center text-sm leading-tight font-black">
                          <FileIcon
                            className="size-4 cursor-pointer text-muted-foreground"
                            onClick={event => {
                              event.stopPropagation();
                            }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="inline-flex shrink grow basis-0 flex-col items-start justify-center">
                      <div className="whitespace-wrap py-1 text-sm leading-normal font-normal break-all">
                        {file.name}
                      </div>
                    </div>
                    <div className="flex size-9 items-center justify-center p-2">
                      <div className="inline-flex size-5 flex-col items-center justify-center gap-2.5">
                        <div className="text-center text-sm leading-tight font-black">
                          <XIcon
                            className="size-4 cursor-pointer text-muted-foreground"
                            onClick={event => {
                              event.stopPropagation();
                              onRemove(index);
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {progress !== 100 && progress !== 0 && (
                <Progress value={progress} className="mt-2 h-1" />
              )}
            </div>
          </div>
        </section>
      )}
    </Dropzone>
  );
};
