import React from "react";
import { Button } from "@/components/ui/button";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTrash } from "@fortawesome/free-solid-svg-icons/faTrash";
import { useDeleteAllDatasets } from "@/api-state/datasets/hooks";
import { useTranslation } from "react-i18next";

interface ClearDatasetsButtonProps {
  onClear?: () => void;
}

export const ClearDatasetsButton: React.FC<ClearDatasetsButtonProps> = ({ onClear }) => {
  const { mutate } = useDeleteAllDatasets();
  const { t } = useTranslation();
  
  const handleClick = () => {
    mutate();
    if (onClear) {
      onClear();
    }
  };
  
  return (
    <Button variant="ghost" onClick={handleClick}>
      <FontAwesomeIcon icon={faTrash} />
      {t("clear_all_datasets")}
    </Button>
  );
};
