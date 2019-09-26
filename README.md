# babypandas
`pandas` data-analysis library with a restricted API

---

Simplified introductory `pandas` library that allows for basic tabular data-analysis with only the necessary methods and arguments. Allows
for a smooth transition into `pandas`.

---

Berkeley `datascience` module equivalents with `babypandas`:

| `datascience` method               		  | `babypandas` equivalent or close                           | method description                        |
|---------------------------------------------|------------------------------------------------------------|-------------------------------------------|
| `Table()`                            		  | `bpd.DataFrame()`                                          | empty table formation                     |
| `Table().with_columns(*labels_and_values)`  | `bpd.DataFrame().assign(**kwargs)`                         | table from lists                          |
| `table.with_columns(*labels_and_values)`    | `df.assign(**kwargs)`                                      | adding columns                            |
| `table.with_rows(rows)`                     | `df.append(other_df, ignore_index=True)`                   |                                           |
| `Table.read_table(filepath)`                | `bpd.read_csv(filepath)`                                   | read in data                              |
| `table.num_columns`                         | `df.shape[1]`                                              | number of columns                         |
| `table.num_rows`                    		  | `df.shape[0]`                                              | number of rows                            |
| `table.labels`                      		  | `df.columns`                                               | list of columns                           |
| `table.relabeled(label, new_label)` 		  | `df.assign(new_label=df.get(label)).drop(columns=[label])` | rename columns                            |
| `table.column(col)`                 		  | `df.get(col)`                                     		   | get a specific column (by name)           |
| `table.column(col).item(0)`         		  | `df.get(col).iloc[0]`                                      | get a specific value in the table         |
| `table.select(col1, col2)`          	      | `df.get([col1, col2])`                                     | get columns as a df                       |
| `table.drop(col1, col2)`            		  | `df.drop(columns=[col1, col2])`                            | drop columns                              |
| `table.sort(col)`                  		  | `df.sort_values(by=col)`                                   | sorts values in a dataframe by col        |
| `table.take(row_indices_or_slice)`          | `df.take(row_indices_or_slice)`                            | selects a single row                      |
| `table.where(col, are.above(num))`   		  | `df.loc[df.get(col) > num]`                                | selects rows based on condition           |
| `table.scatter(xcol, ycol)`                 | `df.plot(kind='scatter', x=xcol, y=ycol)`                  | plots a scatter plot                      |
| `table.plot(xcol, ycol)`                    | `df.plot(x=xcol, y=ycol)`                                  | plots a line plot                         |
| `table.barh(col)`                           | `df.plot(kind='barh', x=col)`                              | plots a horizontal bar plot               |
| `table.hist(col, bins)`                     | `df.get(col).plot(kind='hist', bins=bins)`                 | plots a histogram                         |
| `table.apply(fn, col)`                      | `df.get(col).apply(fn)`                                    | apply function to a column                |
| `table.group(col)`                   		  | `df.groupby(col).count()`                                  | give counts of values in a col            |
| `table.group(col, agg_fn)`           		  | `df.groupby(col).agg_fn.reset_index()`                     | groups by column, aggregates with fn      |
| `table.group([col1, col2])`          		  | `df.groupby([col1, col2]).count().reset_index()`           | groups by two cols, agg with counts       |
| `table.group([col1, col2], sum)`     		  | `df.groupby[col1, col2]).sum().reset_index()`              | groups by two cols, agg with sum          |
| `table.join(leftcol, df2, rightcol)` 		  | `df.merge(df2, left_on=leftcol, right_on=rightcol)`        | merges two dataframes (diff col names)    |
| `table.join(col, df2, col)`          		  | `df.merge(df2, on=col)`                                    | merges two dataframes (same col names)    |
| `table.sample(n)`                    		  | `df.sample(n, replace=True)`                               | sample with replacement                   |
| `sample_proportions(size, distr)`    		  | `np.random.multinomial(size, distr) / size`                | gets sample proportions of a distribution |
