import * as React from 'react';

import { cn } from '~/lib/utils';

type IconProps = React.ComponentProps<React.ElementType>;

type IconPropsWithBehavior<T extends IconProps> = T & {
    behavior: 'append' | 'prepend';
};

type IconComponent<T extends IconProps = IconProps> = React.ComponentType<T>;

export type InputProps<T extends IconComponent = IconComponent> =
    React.InputHTMLAttributes<HTMLInputElement> & {
        icon?: T;
        iconProps: T extends IconComponent<infer P> ? IconPropsWithBehavior<P> : never;
    };

const PromptInput = React.forwardRef<HTMLInputElement, InputProps>(
    (
        {
            className,
            type,
            icon,
            onKeyDown,
            iconProps: { behavior: iconBehavior, ...iconProps },
            ...props
        },
        ref
    ) => {
        const Icon = icon;

        const [isFocused, setIsFocused] = React.useState(false);
        const [isComposing, setIsComposing] = React.useState(false);
        return (
            <div
                className={cn(
                    'border-input file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground aria-invalid:outline-destructive/60 aria-invalid:ring-destructive/20 dark:aria-invalid:outline-destructive dark:aria-invalid:ring-destructive/50 ring-ring/10 dark:ring-ring/20 dark:outline-ring/40 outline-ring/50 aria-invalid:border-destructive/60 dark:aria-invalid:border-destructive h-9 w-full rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium min-w-[600px] min-h-[57px] justify-start items-center inline-flex',
                    isFocused &&
                        'ring-4 outline-1 disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
                    className
                )}
            >
                {Icon && iconBehavior === 'prepend' && (
                    <div className={cn('w-4 h-4 mr-3 text-muted-foreground')}>
                        {<Icon {...iconProps} />}
                    </div>
                )}
                <input
                    type={type}
                    className={cn(
                        'flex items-center justify-center w-full bg-transparent placeholder:text-muted-foreground file:border-0 file:bg-transparent file:text-sm file:font-medium focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50',
                        className
                    )}
                    onCompositionStart={() => setIsComposing(true)}
                    onCompositionEnd={() => setIsComposing(false)}
                    onKeyDown={event => {
                        if (onKeyDown && !isComposing) {
                            onKeyDown(event);
                        }
                    }}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    ref={ref}
                    {...props}
                />
                {Icon && iconBehavior === 'append' && (
                    <div
                        className={cn(
                            'w-4 ml-3 text-muted-foreground h-9 flex-col justify-center items-center inline-flex cursor-pointer'
                        )}
                    >
                        {<Icon {...iconProps} />}
                    </div>
                )}
            </div>
        );
    }
);
PromptInput.displayName = 'PromptInput';

export { PromptInput };
