import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export function isDev() {
    return import.meta.env.MODE === 'development';
}

export function isServedStatic() {
    return window.ENV?.IS_STATIC_FRONTEND;
}
