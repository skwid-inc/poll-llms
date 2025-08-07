non_ordinal_numbers = [
    "oh",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

ordinal_numbers = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "thirtieth",
]

months = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

NUMBER_KEYWORD_BOOST = "&keywords=zero%3A12&keywords=oh%3A12&keywords=one%3A12&keywords=two%3A12&keywords=three%3A12&keywords=four%3A12&keywords=five%3A12&keywords=six%3A12&keywords=seven%3A12&keywords=eight%3A12&keywords=nine%3A12&keywords=ten%3A12&keywords=eleven%3A12&keywords=twelve%3A12&keywords=thirteen%3A12&keywords=fourteen%3A12&keywords=fifteen%3A12&keywords=sixteen%3A12&keywords=seventeen%3A12&keywords=eighteen%3A12&keywords=nineteen%3A12&keywords=twenty%3A12&keywords=thirty%3A12&keywords=fifty%3A12&keywords=sixty%3A12&keywords=seventy%3A12&keywords=eighty%3A12&keywords=ninety%3A12&keywords=first%3A12&keywords=second%3A12&keywords=third%3A12&keywords=fourth%3A12&keywords=fifth%3A12&keywords=sixth%3A12&keywords=seventh%3A12&keywords=eighth%3A12&keywords=ninth%3A12&keywords=tenth%3A12&keywords=eleventh%3A12&keywords=twelfth%3A12&keywords=thirteenth%3A12&keywords=fourteenth%3A12&keywords=fifteenth%3A12&keywords=sixteenth%3A12&keywords=seventeenth%3A12&keywords=eighteenth%3A12&keywords=nineteenth%3A12&keywords=twentieth%3A12&keywords=thirtieth%3A12"
NUMBER_DATE_KEYWORD_BOOST = "&keywords=zero%3A12&keywords=oh%3A12&keywords=one%3A12&keywords=two%3A12&keywords=three%3A12&keywords=four%3A12&keywords=five%3A12&keywords=six%3A12&keywords=seven%3A12&keywords=eight%3A12&keywords=nine%3A12&keywords=ten%3A12&keywords=eleven%3A12&keywords=twelve%3A12&keywords=thirteen%3A12&keywords=fourteen%3A12&keywords=fifteen%3A12&keywords=sixteen%3A12&keywords=seventeen%3A12&keywords=eighteen%3A12&keywords=nineteen%3A12&keywords=twenty%3A12&keywords=thirty%3A12&keywords=fifty%3A12&keywords=sixty%3A12&keywords=seventy%3A12&keywords=eighty%3A12&keywords=ninety%3A12&keywords=first%3A12&keywords=second%3A12&keywords=third%3A12&keywords=fourth%3A12&keywords=fifth%3A12&keywords=sixth%3A12&keywords=seventh%3A12&keywords=eighth%3A12&keywords=ninth%3A12&keywords=tenth%3A12&keywords=eleventh%3A12&keywords=twelfth%3A12&keywords=thirteenth%3A12&keywords=fourteenth%3A12&keywords=fifteenth%3A12&keywords=sixteenth%3A12&keywords=seventeenth%3A12&keywords=eighteenth%3A12&keywords=nineteenth%3A12&keywords=twentieth%3A12&keywords=thirtieth%3A12&keywords=january%3A12&keywords=february%3A12&keywords=march%3A12&keywords=april%3A12&keywords=may%3A12&keywords=june%3A12&keywords=july%3A12&keywords=august%3A12&keywords=september%3A12&keywords=october%3A12&keywords=november%3A12&keywords=december%3A12"
# NAME_KEYWORD_BOOST = "&keywords=mona%3A45&keywords=scottt%3A45&keywords=wynn%3A-20&keywords=len%3A-2&keywords=one%3A-2&keywords=lan%3A-2"


def get_keyword_boost_url_for_names(names, strength=80):
    keyword_boost_strength = strength
    encoded_url = ""
    for name in names:
        subnames = name.split(" ")
        for subname in subnames:
            subname = subname.lower()
            encoded_url += f"&keywords={subname}%3A{keyword_boost_strength}"
    return encoded_url


# encoded_url = ""
# for keyword in non_ordinal_numbers + ordinal_numbers + months:
#     encoded_url += f"&keywords={keyword}%3A12"
# logger.info(encoded_url)

# NUMBER_DATE_KEYWORD_BOOST = encoded_url
