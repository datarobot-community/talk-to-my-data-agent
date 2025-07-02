import React from "react";
import { Button } from "@/components/ui/button";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTrash } from "@fortawesome/free-solid-svg-icons/faTrash";
import { useDeleteAllDatasets } from "@/api/datasets/hooks";

interface ClearDatasetsButtonProps {
  onClear?: () => void;
}

export const ClearDatasetsButton: React.FC<ClearDatasetsButtonProps> = ({
  onClear,
}) => {
  const { mutate } = useDeleteAllDatasets();

  const handleClick = () => {
    mutate();
    if (onClear) {
      onClear();
    }
  };

  return (
    <Button
      testId="clear-datasets-button"
      variant="ghost"
      onClick={handleClick}
    >
      <FontAwesomeIcon icon={faTrash} />
      Clear all datasets
    </Button>
  );
};
