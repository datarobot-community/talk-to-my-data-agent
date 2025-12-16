import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { CheckIcon, ChevronDown } from 'lucide-react';
import { useTranslation } from '@/i18n';
import { cn } from '@/lib/utils';
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

const singleSelectVariants = cva('m-1', {
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

interface SingleSelectProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof singleSelectVariants> {
  options: {
    label: string;
    value: string;
    postfix?: string;
  }[];
  onValueChange: (value: string) => void;
  defaultValue?: string;
  placeholder?: string;
  animation?: number;
  maxCount?: number;
  modalPopover?: boolean;
  asChild?: boolean;
  className?: string;
  testId?: string;
  isLoading?: boolean;
}

export const SingleSelect = React.forwardRef<HTMLButtonElement, SingleSelectProps>(
  (
    {
      options,
      onValueChange,
      variant,
      defaultValue = '',
      placeholder = 'Select options',
      animation = 0,
      modalPopover = false,
      className,
      testId,
      isLoading = false,
      ...props
    },
    ref
  ) => {
    const { t } = useTranslation();
    const [selectedValue, setSelectedValue] = React.useState<string>(defaultValue);
    const [isPopoverOpen, setIsPopoverOpen] = React.useState(false);
    const [isComposing, setIsComposing] = React.useState(false);

    const applyValue = (v: string) => {
      setSelectedValue(v);
      onValueChange(v);
    };

    const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' && !isComposing) {
        setIsPopoverOpen(true);
      } else if (event.key === 'Backspace' && !event.currentTarget.value) {
        applyValue('');
      }
    };

    const handleTogglePopover = () => {
      setIsPopoverOpen(prev => !prev);
    };

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
            {selectedValue ? (
              <div className="flex justify-between items-center w-full">
                <div className="flex flex-wrap items-center">
                  {(() => {
                    const option = options.find(o => o.value === selectedValue);
                    return (
                      <>
                        <Badge
                          type="outline"
                          key={selectedValue}
                          className={cn(singleSelectVariants({ variant }))}
                          style={{ animationDuration: `${animation}s` }}
                        >
                          <TruncatedText>{option?.label}</TruncatedText>
                          {option?.postfix && <span className="ml-1">{option?.postfix}</span>}
                        </Badge>
                      </>
                    );
                  })()}
                </div>
                <div className="flex items-center justify-between">
                  <ChevronDown className="h-4 mx-2 cursor-pointer text-muted-foreground" />
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between w-full mx-auto">
                <span className="text-sm text-muted-foreground mx-3">{placeholder}</span>
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
              {isLoading ? (
                <CommandItem className="flex items-center justify-center py-6">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </CommandItem>
              ) : (
                <>
                  <CommandEmpty>{t('No results found.')}</CommandEmpty>
                  <CommandGroup>
                    {options.map(option => {
                      const isSelected = selectedValue === option.value;
                      return (
                        <CommandItem
                          data-testid={`multi-select-option-${option.value}`}
                          key={option.value}
                          onSelect={() => applyValue(option.value)}
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
                </>
              )}
            </CommandList>
            <CommandSeparator />
            {!!options.length && (
              <CommandGroup>
                <div className="flex items-center justify-between">
                  <CommandItem
                    disabled={isLoading}
                    data-testid="multi-select-close"
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

SingleSelect.displayName = 'SingleSelect';
