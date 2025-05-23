# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""initial migration

Revision ID: 001
Revises:
Create Date: 2024-03-19 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enum types if they don't exist

    # op.execute("CREATE TYPE dataset_type AS ENUM ('standard', 'cleansed', 'dictionary')")
    # op.execute("CREATE TYPE data_source_type AS ENUM ('file', 'database', 'registry', 'generated');")
    # op.execute("CREATE TYPE user_role_type AS ENUM ('user', 'assistant', 'system');")

    # Create dataset_metadata table
    op.create_table(
        'dataset_metadata',
        sa.Column('table_name', sa.String(), nullable=False),
        sa.Column('dataset_type', postgresql.ENUM('standard', 'cleansed', 'dictionary', name='dataset_type'), nullable=False),
        sa.Column('original_name', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('columns', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('data_source', postgresql.ENUM('file', 'database', 'registry', 'generated', name='data_source_type'), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('table_name')
    )

    # Create cleansing_reports table
    op.create_table(
        'cleansing_reports',
        sa.Column('dataset_name', sa.String(), nullable=False),
        sa.Column('report', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['dataset_name'], ['dataset_metadata.table_name'], ),
        sa.PrimaryKeyConstraint('dataset_name')
    )

    # Create chat_history table
    op.create_table(
        'chat_history',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('chat_name', sa.String(), nullable=False),
        sa.Column('data_source', sa.String(), nullable=False, server_default='catalog'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('chat_id', sa.String(), nullable=False),
        sa.Column('role', postgresql.ENUM('user', 'assistant', 'system', name='user_role_type'), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('components', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('in_progress', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['chat_id'], ['chat_history.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    # Drop tables
    op.drop_table('chat_messages')
    op.drop_table('chat_history')
    op.drop_table('cleansing_reports')
    op.drop_table('dataset_metadata')

    # Drop enum types
    op.execute('DROP TYPE user_role_type')
    op.execute('DROP TYPE data_source_type')
    op.execute('DROP TYPE dataset_type')