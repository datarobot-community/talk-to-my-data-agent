import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMagnifyingGlass, faXmark } from '@fortawesome/free-solid-svg-icons';
import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';

interface SearchControlProps {
  onSearch?: (searchText: string) => void;
  placeholder?: string;
  disabled?: boolean;
  testId?: string;
}

export const SearchControl: React.FC<SearchControlProps> = ({
  onSearch,
  disabled = false,
  placeholder,
  testId = 'search-control',
}) => {
  const { t } = useTranslation();
  const placeholderText = placeholder || t('Search');
  const [searchText, setSearchText] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setSearchText(value);
    onSearch?.(value);
  };

  const handleClear = () => {
    setSearchText('');
    setIsExpanded(false);
    onSearch?.('');
  };

  const handleExpand = () => {
    if (!disabled) {
      setIsExpanded(true);
    }
  };

  const handleBlur = () => {
    if (!searchText) {
      setIsExpanded(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      setSearchText('');
      setIsExpanded(false);
      onSearch?.('');
    }
  };

  if (!isExpanded) {
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={handleExpand}
        disabled={disabled}
        className="mr-2 h-9 px-3"
        aria-label={t('Search')}
        data-testid={`${testId}-button`}
      >
        <FontAwesomeIcon icon={faMagnifyingGlass} className="mr-2 size-4" />
        <span className="text-sm">{t('Search')}</span>
      </Button>
    );
  }

  return (
    <div className="mr-2 flex w-48 items-center transition-all duration-300 ease-in-out">
      <FontAwesomeIcon
        icon={faMagnifyingGlass}
        className="mr-3 size-4 shrink-0 text-muted-foreground"
      />
      <div className="relative flex-1">
        {/* Using a native input component because we need custom look and feel here */}
        <input
          type="text"
          value={searchText}
          onChange={handleInputChange}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          autoFocus
          disabled={disabled}
          className="h-9 w-full border-0 border-b border-border bg-transparent pr-8 text-sm transition-colors placeholder:font-medium placeholder:text-muted-foreground focus:border-primary focus:outline-none"
          data-testid={`${testId}-input`}
        />
        {searchText && !disabled && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClear}
            onMouseDown={e => e.preventDefault()}
            className="absolute right-0 top-1/2 size-7 -translate-y-1/2 p-0 hover:bg-muted"
            title={t('Clear')}
            data-testid={`${testId}-clear`}
          >
            <FontAwesomeIcon icon={faXmark} className="size-3" />
          </Button>
        )}
      </div>
    </div>
  );
};
