CREATE TABLE IF NOT EXISTS submissions (
    id SERIAL PRIMARY KEY,
    nickname VARCHAR(50) NOT NULL,
    submission_date DATE NOT NULL,
    players JSONB NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create an index on submission_date for faster queries
CREATE INDEX IF NOT EXISTS idx_submissions_date ON submissions(submission_date);

-- Create a unique constraint to prevent multiple submissions per user per day
CREATE UNIQUE INDEX IF NOT EXISTS idx_submissions_nickname_date ON submissions(nickname, submission_date); 