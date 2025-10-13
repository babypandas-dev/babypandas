# babypandas

A `pandas` data-analysis library with a restricted API

---

The `pandas` library for tabular data analysis is powerful and popular, but perhaps not the easiest to learn: for nearly every
task, no matter how simple, there are multiple ways of approaching
it. `babypandas` is a simplified, introductory `pandas` library that
allows for basic tabular data analysis with only a small subset of
methods and arguments. This restricted interface is designed to be easier to learn while still demonstrating fundamental principles and allowing for a smooth transition into `pandas` at a later time.

The chosen methods are meant to align with the methods in Berkeley's
`datascience` module, developed for the [data8](https://data8.org)
course. However, unlike the `datascience` module, all code written in
`babypandas` is also valid `pandas` code.

---

## Install

To install `babypandas`, use `pip`:

```
pip install babypandas
```

---

## Documentation

See the [documentation](https://babypandas.readthedocs.io) page.

---

## FAQ

_Who is this library for?_

This library is intended for those wanting an introduction to data
science in python, but want a focused, introduction much like what's
covered in Berkeley's data8 course. The pandas methods available in
this library encourage better Pandas usage through functional
programming patterns and method chaining.

_Why not just use the datascience module?_

This library may be prefered over `datascience` when students
will be moving to `pandas`. While this library serves as a restricted
introduction to `pandas`, it doesn't shy away from some `pandas`
usage patterns that may require care for new programmers:

- The frequent use of named function arguments,
- The use of boolean arrays (masks) to select rows,
- The use of table indices.

_How does this library compare to the datascience module?_

Berkeley `datascience` module equivalents with `babypandas`:

| `datascience` method                       | `babypandas` equivalent or close                           | method description                        |
| ------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------- |
| `Table()`                                  | `bpd.DataFrame()`                                          | empty table formation                     |
| `Table().with_columns(*labels_and_values)` | `bpd.DataFrame().assign(**kwargs)`                         | table from lists                          |
| `table.with_columns(*labels_and_values)`   | `df.assign(**kwargs)`                                      | adding columns                            |
| `table.with_rows(rows)`                    | `df.append(other_df, ignore_index=True)`                   |                                           |
| `Table.read_table(filepath)`               | `bpd.read_csv(filepath)`                                   | read in data                              |
| `table.num_columns`                        | `df.shape[1]`                                              | number of columns                         |
| `table.num_rows`                           | `df.shape[0]`                                              | number of rows                            |
| `table.labels`                             | `df.columns`                                               | list of columns                           |
| `table.relabeled(label, new_label)`        | `df.assign(new_label=df.get(label)).drop(columns=[label])` | rename columns                            |
| `table.column(col)`                        | `df.get(col)`                                              | get a specific column (by name)           |
| `table.column(col).item(0)`                | `df.get(col).iloc[0]`                                      | get a specific value in the table         |
| `table.select(col1, col2)`                 | `df.get([col1, col2])`                                     | get columns as a df                       |
| `table.drop(col1, col2)`                   | `df.drop(columns=[col1, col2])`                            | drop columns                              |
| `table.sort(col)`                          | `df.sort_values(by=col)`                                   | sorts values in a dataframe by col        |
| `table.take(row_indices_or_slice)`         | `df.take(row_indices_or_slice)`                            | selects a single row                      |
| `table.where(col, are.above(num))`         | `df.loc[df.get(col) > num]`                                | selects rows based on condition           |
| `table.scatter(xcol, ycol)`                | `df.plot(kind='scatter', x=xcol, y=ycol)`                  | plots a scatter plot                      |
| `table.plot(xcol, ycol)`                   | `df.plot(x=xcol, y=ycol)`                                  | plots a line plot                         |
| `table.barh(col)`                          | `df.plot(kind='barh', x=col)`                              | plots a horizontal bar plot               |
| `table.hist(col, bins)`                    | `df.get(col).plot(kind='hist', bins=bins)`                 | plots a histogram                         |
| `table.apply(fn, col)`                     | `df.get(col).apply(fn)`                                    | apply function to a column                |
| `table.group(col)`                         | `df.groupby(col).count()`                                  | give counts of values in a col            |
| `table.group(col, agg_fn)`                 | `df.groupby(col).agg_fn.reset_index()`                     | groups by column, aggregates with fn      |
| `table.group([col1, col2])`                | `df.groupby([col1, col2]).count().reset_index()`           | groups by two cols, agg with counts       |
| `table.group([col1, col2], sum)`           | `df.groupby[col1, col2]).sum().reset_index()`              | groups by two cols, agg with sum          |
| `table.join(leftcol, df2, rightcol)`       | `df.merge(df2, left_on=leftcol, right_on=rightcol)`        | merges two dataframes (diff col names)    |
| `table.join(col, df2, col)`                | `df.merge(df2, on=col)`                                    | merges two dataframes (same col names)    |
| `table.sample(n)`                          | `df.sample(n, replace=True)`                               | sample with replacement                   |
| `sample_proportions(size, distr)`          | `np.random.multinomial(size, distr) / size`                | gets sample proportions of a distribution |

## Development

Publishing to PyPI requires that a tagged commit exists on the `master` branch. The GitHub Actions workflow will trigger
package building and publishing to PyPI only when a commit on `master` is tagged. This can happen in one of two ways:

1. **Direct Tagged Commit to Master**: Commit your changes directly to `master` and tag the commit before pushing to
   GitHub.

```shell
git commit -m "Your descriptive commit message"
git tag <tag-name> # convention has been to tag with package version
git push origin master
git push origin <tag-name>
```

2. **Merge Pull Request to Master and Post-Hoc Tag**: Merge a pull request into `master`. After merging, tag the resulting
   commit in `master`.

```shell
git checkout master
git pull origin master
git tag <tag-name>
git push origin <tag-name>
```

Either of these approaches will trigger testing, building, and publishing of the package to PyPI.
