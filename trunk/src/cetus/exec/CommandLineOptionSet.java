package cetus.exec;

import java.util.*;

public class CommandLineOptionSet
{
  public final int ANALYSIS = 1;
  public final int TRANSFORM = 2;
  public final int UTILITY = 3;
  public final int CODEGEN = 4;
 
  private class OptionRecord
  {
    public int option_type;
    public String value;
    public String arg;
    public String usage;

    public OptionRecord(int type, String usage)
    {
      this.option_type = type;
      this.value = null;
      this.arg = null;
      this.usage = new String(usage);
    }

    public OptionRecord(int type, String arg, String usage)
    {
      this.option_type = type;
      this.value = null;
      this.arg = new String(arg);
      this.usage = new String(usage);
    }
  }

  private TreeMap<String, OptionRecord> name_to_record;

  public CommandLineOptionSet()
  {
    name_to_record = new TreeMap<String, OptionRecord>();
  }

  public void add(String name, String usage)
  {
    name_to_record.put(name, new OptionRecord(UTILITY, usage));
  }

  public void add(String name, String arg, String usage)
  {
    name_to_record.put(name, new OptionRecord(UTILITY,
						arg, usage));
  }

  public void add(int type, String name, String usage)
  {
    name_to_record.put(name, new OptionRecord(type, usage));
  }

  public void add(int type, String name, String arg, String usage)
  {
    name_to_record.put(name, new OptionRecord(type, arg, usage));
  }

  public boolean contains(String name)
  {
    return name_to_record.containsKey(name);
  }

  public String getUsage()
  {
    String usage = new String();

    usage += "UTILITY\n------------------------------------------------------\n";
    usage += getUsage(UTILITY);
    usage += "ANALYSIS\n------------------------------------------------------\n";
    usage += getUsage(ANALYSIS);
    usage += "TRANSFORM\n------------------------------------------------------\n";
    usage += getUsage(TRANSFORM);
    usage += "CODEGEN\n------------------------------------------------------\n";
    usage += getUsage(CODEGEN);

    return usage;
  }

  public String getUsage(int type)
  {
    String usage = new String();

    Iterator<Map.Entry<String, OptionRecord>> iter
      = name_to_record.entrySet().iterator();

    while (iter.hasNext())
    {
      Map.Entry<String, OptionRecord> pair = iter.next();
      OptionRecord record = pair.getValue();

      if (record.option_type == type)
      {
        usage += "-"; 
        usage += pair.getKey();


        if (record.arg != null)
        {
          usage += "=";
          usage += record.arg;
        }

        usage += "\n    ";
        usage += record.usage;
        usage += "\n";
      }
    }

    return usage;
  }

  public String getValue(String name)
  {
    OptionRecord record = name_to_record.get(name);

    if (record == null)
		/*
      return new String();
		*/
			return null;
    else
      return record.value;
  }

  public void setValue(String name, String value)
  {
    OptionRecord record = name_to_record.get(name);

    if (record != null)
      record.value = value;
  }

  public int getType(String name)
  {
    OptionRecord record = name_to_record.get(name);

    if (record == null)
      return 0;
    else
      return record.option_type;
  }
}
