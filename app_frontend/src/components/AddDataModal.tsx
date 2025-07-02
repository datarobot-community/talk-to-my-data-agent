import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus } from "@fortawesome/free-solid-svg-icons/faPlus";
import { DataSourceSelector } from "./DataSourceSelector";
import { DATA_SOURCES } from "@/constants/dataSources";
import { MultiSelect } from "@/components/ui-custom/multi-select";
import { useState } from "react";
import { FileUploader } from "./ui-custom/file-uploader";
import { useFetchAllDatasets } from "@/api/datasets/hooks";
import {
  useGetDatabaseTables,
  useLoadFromDatabaseMutation,
} from "@/api/database/hooks";
import { useFileUploadMutation, UploadError } from "@/api/datasets/hooks";
import { Separator } from "@radix-ui/react-separator";
import loader from "@/assets/loader.svg";
import { useAppState } from "@/state/hooks";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AxiosError } from "axios";
import { TruncatedText } from "./ui-custom/truncated-text";

export const AddDataModal = () => {
  const { data } = useFetchAllDatasets();
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const { data: dbTables } = useGetDatabaseTables();
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const { setDataSource, dataSource } = useAppState();
  const [files, setFiles] = useState<File[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { mutate, progress } = useFileUploadMutation({
    onSuccess: () => {
      setIsPending(false);
      setError(null);
      setIsOpen(false);
    },
    onError: (error: UploadError | AxiosError) => {
      setIsPending(false);
      console.error(error);
      setError(error.message || "An error occurred while uploading files");
    },
  });

  const { mutate: loadFromDatabase } = useLoadFromDatabaseMutation({
    onSuccess: () => {
      setIsPending(false);
      setIsOpen(false);
    },
    onError: (error: Error) => {
      setIsPending(false);
      console.error(error);
    },
  });

  return (
    <Dialog
      defaultOpen={isOpen}
      onOpenChange={(open) => {
        setIsOpen(open);
        setError(null);
        setFiles([]);
      }}
      open={isOpen}
    >
      <DialogTrigger asChild>
        <Button variant="outline" testId="add-data-button">
          <FontAwesomeIcon icon={faPlus} /> Add Data
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Add Data</DialogTitle>
          <Separator className="border-t" />
          <DialogDescription></DialogDescription>
        </DialogHeader>
        <DataSourceSelector value={dataSource} onChange={setDataSource} />
        <Separator className="my-4 border-t" />
        {dataSource == DATA_SOURCES.FILE && (
          <>
            <div className="h-10 flex-col justify-start items-start inline-flex">
              <div className="text-primary text-sm font-semibold leading-normal">
                Local files
              </div>
              <div className="text-muted-foreground text-sm font-normal leading-normal">
                Select one or more CSV, XLSX, XLS files, up to 200MB.
              </div>
            </div>
            <FileUploader onFilesChange={setFiles} progress={progress} />
            <h4>Data Registry</h4>
            <h6>Select one or more catalog items</h6>
            <MultiSelect
              options={
                data
                  ? data.map((i) => ({
                      label: i.name,
                      value: i.id,
                      postfix: i.size,
                    }))
                  : []
              }
              onValueChange={setSelectedDatasets}
              defaultValue={selectedDatasets}
              placeholder="Select one or more items."
              variant="inverted"
              modalPopover
              animation={2}
              maxCount={3}
            />
            {error && (
              <Alert variant="destructive">
                <AlertDescription>
                  <TruncatedText maxLength={100}>{error}</TruncatedText>
                </AlertDescription>
              </Alert>
            )}
          </>
        )}

        {dataSource == DATA_SOURCES.DATABASE && (
          <>
            <h4>Databases</h4>
            <h6>Select one or more tables</h6>
            <MultiSelect
              options={
                dbTables
                  ? dbTables.map((i) => ({
                      label: i,
                      value: i,
                    }))
                  : []
              }
              onValueChange={setSelectedTables}
              defaultValue={selectedTables}
              placeholder="Select one or more items."
              variant="inverted"
              testId="database-table-select"
              modalPopover
              animation={2}
              maxCount={3}
            />
          </>
        )}
        <Separator className="border-t mt-6" />
        <DialogFooter>
          <Button variant={"ghost"} onClick={() => setIsOpen(false)}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="secondary"
            disabled={isPending}
            testId="add-data-modal-save-button"
            onClick={() => {
              setError(null);
              setIsPending(true);
              if (dataSource === DATA_SOURCES.DATABASE) {
                if (selectedTables.length > 0) {
                  loadFromDatabase({ tableNames: selectedTables });
                }
              } else {
                mutate({ files, catalogIds: selectedDatasets });
              }
            }}
          >
            {isPending && (
              <img
                src={loader}
                alt="downloading"
                className="w-4 h-4 animate-spin"
              />
            )}
            Save selections
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
