# Data Analyst React Frontend

This application provides a modern React-based frontend for the **Talk to My Data** application. It allows users to interact with data, perform analyses, and chat with the system to gain insights from their datasets.

## Features

- Interactive chat interface for data analysis
- Data visualization with interactive plots
- Dataset management and cleansing
- Support for multiple data sources (CSV, Data Registry, Snowflake, Google Cloud)
- Code execution and insights generation
- Multi-language support (i18n) for English and Japanese

## Tech Stack

- React 18 with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- Jest for testing
- React Query for API state management
- i18next & react-i18next for internationalization

## Internationalization (i18n)

This project supports multiple languages (currently English and Japanese) using [i18next](https://www.i18next.com/) and [react-i18next](https://react.i18next.com/).

### How it works

- All visible UI text is managed via translation keys and loaded from JSON files in `src/locales/en/translation.json` (English) and `src/locales/ja/translation.json` (Japanese).
- The language can be switched at any time using the flag button in the sidebar (top right, next to the DataRobot logo).
- The current language is applied instantly across all pages and components.

### Adding or updating translations

- To update text, edit the corresponding key in the translation JSON files.
- To add a new language:
  1. Create a new folder in `src/locales/` (e.g., `fr` for French).
  2. Add a `translation.json` file with the required keys and translations.
  3. Register the new language in `src/i18n.ts`.

### Using translations in code

- Use the `useTranslation` hook from `react-i18next` in your component:
  ```tsx
  import { useTranslation } from "react-i18next";
  const { t } = useTranslation();
  // ...
  <span>{t("your_translation_key")}</span>
  ```
- Never hardcode visible UI text; always use translation keys.

### Example

```json
// src/locales/en/translation.json
{
  "add_data": "Add Data",
  "delete_chat": "Delete chat"
}
```

```tsx
// In a React component
import { useTranslation } from "react-i18next";
const { t } = useTranslation();
<Button>{t("add_data")}</Button>
```

## Development

To start the development server (cd into the `frontend_react/react_src` directory first):

```bash
npm i
npm run dev
```

To start backend:

in project root

```bash
uvicorn utils.rest_api:app --port 8080
```

## Building

To build the application for production:

```bash
npm run build
```

The build output will be placed in the `../deploy/dist` directory, which is then used by the Python backend to serve the application. When using the React frontend through the `FRONTEND_TYPE="react"` environment variable, the application will look for the built files in this location.

## Testing

To run the test suite:

```bash
npm run test
```

## Project Structure

- `src/api-state`: API client and hooks for data fetching
- `src/components/ui`: shadcn components
- `src/components/ui-custom`: shadcn based generic components
- `src/pages`: Main application pages
- `src/state`: Application state management
- `src/assets`: Static assets like images and icons
