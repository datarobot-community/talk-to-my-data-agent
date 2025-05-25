import React, { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { DictionaryTable as DT, DictionaryRow } from "@/api-state/dictionaries/types";
import { Input } from "@/components/ui/input";
import { useTranslation } from "react-i18next";

interface DictionaryTableProps {
  data: DT;
  onUpdateCell?: (rowIndex: number, field: keyof DictionaryRow, value: string) => void;
}

interface EditableCellProps {
  initialValue: string;
  rowIndex: number;
  field: keyof DictionaryRow;
  onUpdate?: (rowIndex: number, field: keyof DictionaryRow, value: string) => void;
}

const EditableCell: React.FC<EditableCellProps> = ({ 
  initialValue, 
  rowIndex, 
  field, 
  onUpdate 
}) => {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(initialValue);
  const { t } = useTranslation();
  
  // Update local state when props change (e.g., when data is refreshed after an error)
  React.useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const handleDoubleClick = () => {
    setEditing(true);
  };

  const handleBlur = () => {
    setEditing(false);
    if (value !== initialValue && onUpdate) {
      onUpdate(rowIndex, field, value);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      setEditing(false);
      if (value !== initialValue && onUpdate) {
        onUpdate(rowIndex, field, value);
      }
    } else if (e.key === "Escape") {
      setEditing(false);
      setValue(initialValue);
    }
  };

  return editing ? (
    <Input
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoFocus
      className="w-full h-[28px]"
    />
  ) : (
    <div 
      onDoubleClick={handleDoubleClick}
      className="cursor-pointer hover:bg-secondary p-1 rounded min-h-[28px]"
      title={t("double_click_edit")}
    >
      {value}
    </div>
  );
};

export const DictionaryTable: React.FC<DictionaryTableProps> = ({ 
  data, 
  onUpdateCell 
}) => {
  const { t } = useTranslation();
  const handleCellUpdate = (rowIndex: number, field: keyof DictionaryRow, value: string) => {
    if (onUpdateCell) {
      onUpdateCell(rowIndex, field, value);
    }
  };

  return (
    <Table>
      <TableHeader className="bg-background">
        <TableRow>
          <TableHead className="w-[200px]">{t("column")}</TableHead>
          <TableHead>{t("type")}</TableHead>
          <TableHead>{t("description_col")}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.column_descriptions?.map((column, index) => (
          <TableRow key={column.column}>
            <TableCell className="font-medium">{column.column}</TableCell>
            <TableCell>
              <EditableCell 
                initialValue={column.data_type} 
                rowIndex={index} 
                field="data_type" 
                onUpdate={handleCellUpdate} 
              />
            </TableCell>
            <TableCell>
              <EditableCell 
                initialValue={column.description} 
                rowIndex={index} 
                field="description" 
                onUpdate={handleCellUpdate} 
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};
