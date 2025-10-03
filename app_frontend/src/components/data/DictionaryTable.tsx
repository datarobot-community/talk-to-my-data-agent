import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useTranslation } from '@/i18n';
import { DictionaryTable as DT, DictionaryRow } from '@/api/dictionaries/types';
import { Input } from '@/components/ui/input';
import { HighlightText } from '@/components/ui-custom/highlight-text';

interface DictionaryTableProps {
  data: DT;
  onUpdateCell?: (rowIndex: number, field: keyof DictionaryRow, value: string) => void;
  searchText?: string;
}

interface EditableCellProps {
  initialValue: string;
  rowIndex: number;
  field: keyof DictionaryRow;
  onUpdate?: (rowIndex: number, field: keyof DictionaryRow, value: string) => void;
  searchText?: string;
}

const EditableCell: React.FC<EditableCellProps> = ({
  initialValue,
  rowIndex,
  field,
  onUpdate,
  searchText,
}) => {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(initialValue);

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
    if (e.key === 'Enter') {
      e.preventDefault();
      setEditing(false);
      if (value !== initialValue && onUpdate) {
        onUpdate(rowIndex, field, value);
      }
    } else if (e.key === 'Escape') {
      setEditing(false);
      setValue(initialValue);
    }
  };

  return editing ? (
    <Input
      value={value}
      onChange={e => setValue(e.target.value)}
      onBlur={handleBlur}
      onKeyDown={handleKeyDown}
      autoFocus
      className="w-full h-[28px]"
    />
  ) : (
    <div
      onDoubleClick={handleDoubleClick}
      className="cursor-pointer hover:bg-secondary p-1 rounded min-h-[28px] text-muted-foreground"
      title="Double-click to edit"
    >
      <HighlightText text={value} searchText={searchText || ''} />
    </div>
  );
};

function displayDataType(dataType: string): string {
  const match = dataType.match(/(\w+)\(/);
  if (match) {
    return match[1];
  }
  return dataType;
}

export const DictionaryTable: React.FC<DictionaryTableProps> = ({
  data,
  onUpdateCell,
  searchText,
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
          <TableHead className="w-[200px]">{t('Column')}</TableHead>
          <TableHead>{t('Type')}</TableHead>
          <TableHead>{t('Description')}</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data.column_descriptions?.map((column, index) => (
          <TableRow key={column.column}>
            <TableCell className="font-medium">
              <HighlightText text={column.column} searchText={searchText || ''} />
            </TableCell>
            <TableCell>
              <EditableCell
                initialValue={displayDataType(column.data_type)}
                rowIndex={index}
                field="data_type"
                onUpdate={handleCellUpdate}
                searchText={searchText}
              />
            </TableCell>
            <TableCell>
              <EditableCell
                initialValue={column.description}
                rowIndex={index}
                field="description"
                onUpdate={handleCellUpdate}
                searchText={searchText}
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
};
