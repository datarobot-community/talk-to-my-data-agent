import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { CheckIcon, ChevronDown, XIcon } from 'lucide-react';
import { useTranslation } from '@/i18n';
import { cn } from '@/lib/utils';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2 } from 'lucide-react';
import { TruncatedText } from '@/components/ui-custom/truncated-text';

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from '@/components/ui/command';

const multiSelectVariants = cva('m-1', {
  variants: {
    variant: {
      default: 'border-foreground/10 text-foreground bg-card hover:bg-card/80',
      secondary:
        'border-foreground/10 bg-secondary text-secondary-foreground hover:bg-secondary/80',
      destructive:
        'border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80',
      inverted: 'inverted',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
});

interface MultiSelectProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof multiSelectVariants> {
  options: {
    label: string;
    value: string;
    postfix?: string;
  }[];
  onValueChange: (value: string[]) => void;
  defaultValue?: string[];
  placeholder?: string;
  animation?: number;
  maxCount?: number;
  modalPopover?: boolean;
  asChild?: boolean;
  className?: string;
  testId?: string;
  isLoading?: boolean;
}

export const MultiSelect = React.forwardRef<HTMLButtonElement, MultiSelectProps>(
  (
    {
      options,
      onValueChange,
      variant,
      defaultValue = [],
      placeholder = 'Select options',
      animation = 0,
      maxCount = 3,
      isLoading = false,
      modalPopover = false,
      className,
      testId,
      ...props
    },
    ref
  ) => {
    const { t } = useTranslation();
    const [selectedValues, setSelectedValues] = React.useState<string[]>(defaultValue);
    const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
    const [isComposing, setIsComposing] = React.useState(false);

    const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' && !isComposing) {
        setIsPopoverOpen(true);
      } else if (event.key === 'Backspace' && !event.currentTarget.value) {
        toggleOption(selectedValues[selectedValues.length - 1]);
      }
    };

    const toggleOption = (optionValue?: string) => {
      if (optionValue) {
        const newSelectedValues = selectedValues.includes(optionValue)
          ? selectedValues.filter(value => value !== optionValue)
          : [...selectedValues, optionValue];
        setSelectedValues(newSelectedValues);
        onValueChange(newSelectedValues);
      }
    };

    const handleClear = () => {
      setSelectedValues([]);
      onValueChange([]);
    };

    const handleTogglePopover = () => {
      setIsPopoverOpen(prev => !prev);
    };

    const clearExtraOptions = () => {
      const trimmedValues = selectedValues.slice(0, maxCount);
      setSelectedValues(trimmedValues);
      onValueChange(trimmedValues);
    };

    React.useEffect(() => setSelectedValues(defaultValue), [defaultValue]);

    return (
      <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen} modal={modalPopover}>
        <PopoverTrigger asChild>
          <Button
            ref={ref}
            {...props}
            onClick={handleTogglePopover}
            className={cn(
              'flex w-full p-1 rounded-md border min-h-10 h-auto items-center justify-between bg-inherit hover:bg-inherit [&_svg]:pointer-events-auto',
              className
            )}
            testId={testId}
          >
            {selectedValues.length > 0 ? (
              <div className="flex justify-between items-center w-full">
                <div className="flex flex-wrap items-center">
                  {selectedValues.slice(0, maxCount).map(value => {
                    const option = options.find(o => o.value === value);
                    return (
                      <Badge
                        key={value}
                        className={cn(multiSelectVariants({ variant }))}
                        style={{ animationDuration: `${animation}s` }}
                        type="outline"
                      >
                        <TruncatedText>{option?.label}</TruncatedText>
                        {option?.postfix && <span className="ml-1">{option?.postfix}</span>}
                        <XIcon
                          className="ml-2 h-4 w-4 cursor-pointer transition ease-in-out delay-150 hover:scale-120 duration-300"
                          onClick={event => {
                            event.stopPropagation();
                            toggleOption(value);
                          }}
                        />
                      </Badge>
                    );
                  })}
                  {selectedValues.length > maxCount && (
                    <Badge
                      className={cn(
                        'bg-transparent text-foreground border-foreground/1 hover:bg-transparent',
                        multiSelectVariants({ variant })
                      )}
                      type="outline"
                      style={{ animationDuration: `${animation}s` }}
                    >
                      {`+ ${selectedValues.length - maxCount} more`}
                      <XIcon
                        className="ml-2 h-4 w-4 cursor-pointer transition ease-in-out delay-150 hover:scale-120 duration-300"
                        onClick={event => {
                          event.stopPropagation();
                          clearExtraOptions();
                        }}
                      />
                    </Badge>
                  )}
                </div>
                <div className="flex items-center justify-between">
                  <XIcon
                    className="h-4 mx-2 cursor-pointer text-muted-foreground"
                    onClick={event => {
                      event.stopPropagation();
                      handleClear();
                    }}
                  />
                  <Separator orientation="vertical" className="flex min-h-6 h-full" />
                  <ChevronDown className="h-4 mx-2 cursor-pointer text-muted-foreground" />
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between w-full mx-auto">
                <span className="body-secondary mx-3">{placeholder}</span>
                <ChevronDown className="h-4 cursor-pointer text-muted-foreground mx-2" />
              </div>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent
          className="w-auto p-0"
          align="start"
          onEscapeKeyDown={() => setIsPopoverOpen(false)}
        >
          <Command>
            <CommandInput
              placeholder={t('Search...')}
              disabled={isLoading || !options.length}
              onKeyDown={handleInputKeyDown}
              onCompositionStart={() => setIsComposing(true)}
              onCompositionEnd={() => setIsComposing(false)}
            />
            <CommandList className="max-w-[800px]">
              <CommandEmpty>{t('No results found.')}</CommandEmpty>
              {isLoading ? (
                <CommandItem className="flex items-center justify-center py-6">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </CommandItem>
              ) : (
                <CommandGroup>
                  {options.map(option => {
                    const isSelected = selectedValues.includes(option.value);
                    return (
                      <CommandItem
                        data-testid={`multi-select-option-${option.value}`}
                        key={option.value}
                        onSelect={() => toggleOption(option.value)}
                        className="cursor-pointer"
                      >
                        <div
                          className={cn(
                            'mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary',
                            {
                              'opacity-50 [&_svg]:invisible': !isSelected,
                            }
                          )}
                        >
                          <CheckIcon className="h-4 w-4 text-primary" />
                        </div>
                        <span>{option.label}</span>
                        {option?.postfix && (
                          <span className="ml-1 grow text-right">{option?.postfix}</span>
                        )}
                      </CommandItem>
                    );
                  })}
                </CommandGroup>
              )}
            </CommandList>
            <CommandSeparator />
            {!!options.length && (
              <CommandGroup>
                <div className="flex items-center justify-between">
                  {selectedValues.length > 0 && (
                    <>
                      <CommandItem
                        onSelect={handleClear}
                        className="flex-1 justify-center cursor-pointer"
                      >
                        {t('Clear')}
                      </CommandItem>
                      <Separator orientation="vertical" className="flex min-h-6 h-full" />
                    </>
                  )}
                  <CommandItem
                    data-testid="multi-select-close"
                    disabled={isLoading}
                    onSelect={() => setIsPopoverOpen(false)}
                    className="flex-1 justify-center cursor-pointer max-w-full"
                  >
                    {t('Confirm')}
                  </CommandItem>
                </div>
              </CommandGroup>
            )}
          </Command>
        </PopoverContent>
      </Popover>
    );
  }
);

MultiSelect.displayName = 'MultiSelect';
