import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons/faMagnifyingGlass';

interface SearchControlProps {
    onSearch?: (searchText: string) => void;
}

export const SearchControl: React.FC<SearchControlProps> = () => {
    return (
        <div>
            <FontAwesomeIcon className="mx-2" icon={faMagnifyingGlass} />
            Search
        </div>
    );
};
