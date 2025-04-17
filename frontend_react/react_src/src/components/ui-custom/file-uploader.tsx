import React, { useState } from "react";
import Dropzone, { type FileRejection } from "react-dropzone";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faFolder } from "@fortawesome/free-solid-svg-icons/faFolder";
import { XIcon, FileIcon } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface FileUploaderProps {
  maxSize?: number;
  accept?: { [key: string]: string[] };
  onFilesChange: (files: File[]) => void;
  progress: number;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  maxSize = 1024 * 1024 * 200,
  accept = { "file/csv": [".csv"] },
  progress = 0,
  onFilesChange,
}) => {
  const [files, setFiles] = useState<File[]>([]);

  const onDrop = React.useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      const newFiles = acceptedFiles.map((file) =>
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
            className="border border-dashed border-primary/20 p-4 rounded-lg cu"
          >
            <input {...getInputProps()} />
            <p className="text-center p-6">
              Drag and drop from your desktop, or{" "}
              <FontAwesomeIcon icon={faFolder} />{" "}
              <strong>browse local files</strong>
            </p>
            <div>
              {files.map((file, index) => (
                <div
                  key={index}
                  className="h-min-[52px] pt-4 flex-col justify-start items-start gap-2.5 flex"
                >
                  <div className="self-stretch h-9 bg-secondary/50 rounded border border-primary/10 justify-start items-center gap-2 inline-flex">
                    <div className="w-9 self-stretch p-2 bg-secondary rounded-tl-[3px] rounded-bl-[3px] flex-col justify-center items-center gap-2 inline-flex">
                      <div className="w-9 h-9 flex-col justify-center items-center gap-2.5 flex">
                        <div className="text-center text-sm font-black leading-tight">
                          <FileIcon
                            className="w-4 h-4 cursor-pointer text-muted-foreground"
                            onClick={(event) => {
                              event.stopPropagation();
                            }}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="grow shrink basis-0 flex-col justify-center items-start inline-flex">
                      <div className="text-sm font-normal leading-tight">
                        {file.name}
                      </div>
                    </div>
                    <div className="w-9 h-9 p-2 justify-center items-center flex">
                      <div className="w-5 h-5 flex-col justify-center items-center gap-2.5 inline-flex">
                        <div className="text-center text-sm font-black leading-tight">
                          <XIcon
                            className="w-4 h-4 cursor-pointer text-muted-foreground"
                            onClick={(event) => {
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
                <Progress value={progress} className="h-1 mt-2" />
              )}
            </div>
          </div>
        </section>
      )}
    </Dropzone>
  );
};
